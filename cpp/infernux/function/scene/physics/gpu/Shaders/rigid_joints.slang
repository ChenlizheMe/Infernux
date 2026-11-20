// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_RIGID_JOINTS_SLANG
#define JPH_GPU_RIGID_JOINTS_SLANG

#include "rigid_math.slang"
#include "rigid_psd.slang"

RWStructuredBuffer<QueryRecordWord> joint_stream;

// Header: count, author-end word, total words, zero. Each immutable
// directory is exactly three words; payload extents come from the shared schema.
JointDirectory jph_joint_load_directory(uint index)
{
    uint p = 1u + JointDirectoryWords * index;
    JointDirectory directory;
    directory.bodies = joint_stream[p].value;
    directory.identity = joint_stream[p + 1u].value;
    directory.layout = joint_stream[p + 2u].value;
    return directory;
}

uint jph_joint_state_words(uint kind)
{
    switch (kind)
    {
    case JointFixed: return JointFixedStateWords;
    case JointHinge: return JointHingeStateWords;
    case JointSlider: return JointSliderStateWords;
    case JointDistance: return JointDistanceStateWords;
    case JointSwingTwist: return JointSwingStateWords;
    case JointSixDOF: return JointSixStateWords;
    default: return 0u;
    }
}

void jph_joint_load_author(JointDirectory directory, out JointFixedAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_FIXED_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointFixedState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_FIXED_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_store_state(JointDirectory directory, JointFixedState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_FIXED_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

void jph_joint_load_author(JointDirectory directory, out JointHingeAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_HINGE_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointHingeState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_HINGE_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_store_state(JointDirectory directory, JointHingeState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_HINGE_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

void jph_joint_load_author(JointDirectory directory, out JointSliderAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_SLIDER_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointSliderState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_SLIDER_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_store_state(JointDirectory directory, JointSliderState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_SLIDER_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

void jph_joint_load_author(JointDirectory directory, out JointDistanceAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_DISTANCE_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointDistanceState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_DISTANCE_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_store_state(JointDirectory directory, JointDistanceState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_DISTANCE_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

void jph_joint_load_author(JointDirectory directory, out JointSwingAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_SWING_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_author(JointDirectory directory, out JointSixAuthor a)
{
    a.bodies = directory.bodies;
    a.identity = directory.identity;
    uint p = directory.layout.x;
#define JPH_LOAD_F(name) a.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) a.name = joint_stream[p++].value;
    JPH_SIX_AUTHOR_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointSwingState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_SWING_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_load_state(JointDirectory directory, out JointSixState s)
{
    uint p = directory.layout.y;
#define JPH_LOAD_F(name) s.name = asfloat(joint_stream[p++].value);
#define JPH_LOAD_U(name) s.name = joint_stream[p++].value;
    JPH_SIX_STATE_FIELDS(JPH_LOAD_F,JPH_LOAD_U)
#undef JPH_LOAD_F
#undef JPH_LOAD_U
}

void jph_joint_store_state(JointDirectory directory, JointSwingState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_SWING_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

void jph_joint_store_state(JointDirectory directory, JointSixState s)
{
    uint p = directory.layout.y;
#define JPH_STORE_F(name) joint_stream[p++].value = asuint(s.name);
#define JPH_STORE_U(name) joint_stream[p++].value = s.name;
    JPH_SIX_STATE_FIELDS(JPH_STORE_F,JPH_STORE_U)
#undef JPH_STORE_F
#undef JPH_STORE_U
}

float4 jph_joint_conjugate(float4 q) { return float4(-q.xyz, q.w); }

// Six-only SO(3) cut convention. Old joint drivers never call this helper.
// At exact pi the dominant vector component decides sign; strict comparisons
// retain deterministic x/y/z priority when magnitudes tie.
float4 jph_six_cut(float4 q)
{
    if (q.w < 0.0f) return -q;
    if (q.w > 0.0f) return q;
    uint dominant = 0u;
    if (abs(q.y) > abs(q.x)) dominant = 1u;
    if (abs(q.z) > abs(q[dominant])) dominant = 2u;
    if (q[dominant] < 0.0f) q = -q;
    q.w = 0.0f; // Signed zero is the same exact cut, including atan2 inputs.
    return q;
}

BodyState jph_joint_world_state()
{
    BodyState s;
    s.positionCOM = float4(0.0f);
    s.rotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    s.linearVelocity = float4(0.0f);
    s.angularVelocity = float4(0.0f);
    return s;
}

BodyModel jph_joint_world_model()
{
    BodyModel m;
    m.meta = uint4(0u);
    m.mass = float4(0.0f);
    m.invInertiaDiagonal = float4(0.0f);
    m.inertiaRotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    m.damping = float4(0.0f);
    return m;
}

float3 jph_joint_perpendicular(float3 v)
{
    // Vec3::GetNormalizedPerpendicular, including the same deterministic basis.
    return normalize(abs(v.x) > abs(v.y) ? float3(v.z, 0.0f, -v.x) : float3(0.0f, v.z, -v.y));
}

float4 jph_joint_frame(float3 x, float3 y, float3 z)
{
    // Quaternion from rotation columns, avoiding a poorly conditioned trace
    // branch near PI. Frames supplied here are orthonormal.
    float4 q;
    float trace = x.x + y.y + z.z;
    if (trace > 0.0f)
    {
        float t = 2.0f * sqrt(1.0f + trace);
        q = float4((y.z - z.y) / t, (z.x - x.z) / t, (x.y - y.x) / t, 0.25f * t);
    }
    else if (x.x > y.y && x.x > z.z)
    {
        float t = 2.0f * sqrt(1.0f + x.x - y.y - z.z);
        q = float4(0.25f * t, (y.x + x.y) / t, (z.x + x.z) / t, (y.z - z.y) / t);
    }
    else if (y.y > z.z)
    {
        float t = 2.0f * sqrt(1.0f + y.y - x.x - z.z);
        q = float4((y.x + x.y) / t, 0.25f * t, (z.y + y.z) / t, (z.x - x.z) / t);
    }
    else
    {
        float t = 2.0f * sqrt(1.0f + z.z - x.x - y.y);
        q = float4((z.x + x.z) / t, (z.y + y.z) / t, 0.25f * t, (x.y - y.x) / t);
    }
    return normalize(q);
}

void jph_joint_unscale(float3 offset, float3 scale, float3 raw,
                       out float4 anchor, out float4 collapsed)
{
    anchor = float4(raw, 0.0f);
    collapsed = float4(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis)
        if (scale[axis] != 0.0f)
            anchor[axis] = offset[axis] / scale[axis];
        else
            collapsed[axis] = offset[axis];
}

JointFixedState jph_joint_initialize(JointFixedAuthor a, BodyState s1, BodyState s2)
{
    JointFixedState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        float3 raw = float3(0.0f);
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);

    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    s.frame2 = normalize(jph_quat_mul(jph_joint_conjugate(s2.rotation), world_frame));
    s.pointImpulse = float4(0.0f);
    s.angularImpulse = float4(0.0f);

    if (!(all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.frame1))
        && all(isfinite(s.frame2))
        && all(isfinite(s.pointImpulse))
        && all(isfinite(s.angularImpulse))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

JointHingeState jph_joint_initialize(JointHingeAuthor a, BodyState s1, BodyState s2)
{
    JointHingeState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    s.poseDriveRelative = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        float3 raw = float3(0.0f);
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);
    {
        float3 z = normalize(jph_rotate(s2.rotation, a.ownerAxis.xyz));
        float3 x = jph_joint_perpendicular(z);
        world_frame = jph_joint_frame(x, cross(z, x), z);
    }

    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    s.frame2 = normalize(jph_quat_mul(jph_joint_conjugate(s2.rotation), world_frame));
    s.pointImpulse = float4(0.0f);
    s.angularImpulse = float4(0.0f);
    s.axisImpulse = float4(0.0f);
    s.rotor = float4(0.0f, 0.0f, 0.0f, a.passive.z);
    s.setup = float4(0.0f);

    {
        float3 axis = jph_rotate(jph_quat_mul(s1.rotation, s.frame1), float3(0.0f, 0.0f, 1.0f));
        s.rotor.x = dot(axis, s2.angularVelocity.xyz - s1.angularVelocity.xyz);
    }
    if (!(all(isfinite(s.poseDriveRelative))
        && all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.frame1))
        && all(isfinite(s.frame2))
        && all(isfinite(s.pointImpulse))
        && all(isfinite(s.angularImpulse))
        && all(isfinite(s.axisImpulse))
        && all(isfinite(s.rotor))
        && all(isfinite(s.setup))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

JointSliderState jph_joint_initialize(JointSliderAuthor a, BodyState s1, BodyState s2)
{
    JointSliderState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        float3 raw = float3(0.0f);
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);

    {
        float3 x = normalize(jph_rotate(s2.rotation, a.ownerAxis.xyz));
        float3 y = jph_joint_perpendicular(x);
        world_frame = jph_joint_frame(x, y, cross(x, y));
    }

    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    s.frame2 = normalize(jph_quat_mul(jph_joint_conjugate(s2.rotation), world_frame));
    s.pointImpulse = float4(0.0f);
    s.angularImpulse = float4(0.0f);
    s.axisImpulse = float4(0.0f);

    s.setup = float4(0.0f);

    if (!(all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.frame1))
        && all(isfinite(s.frame2))
        && all(isfinite(s.pointImpulse))
        && all(isfinite(s.angularImpulse))
        && all(isfinite(s.axisImpulse))
        && all(isfinite(s.setup))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

JointDistanceState jph_joint_initialize(JointDistanceAuthor a, BodyState s1, BodyState s2)
{
    JointDistanceState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = a.connectedAnchor;
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = a.connectedAnchor.xyz;
        float3 raw = a.connectedLocalAnchor.xyz;
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);

    s.axisImpulse = float4(0.0f);

    s.setup = float4(0.0f);
    s.distanceReference = float4(0.0f);

    {
        float3 connected_point = a.bodies.x == InvalidQueryBodySlot ? a.connectedAnchor.xyz
            : s1.positionCOM.xyz + jph_rotate(s1.rotation, a.connectedAnchor.xyz - a.rootCOM1.xyz);
        // Jolt retains +Y until the first positive-length properties calculation.
        s.distanceReference = float4(0.0f, 1.0f, 0.0f, length(world_anchor - connected_point));
    }

    if (!(all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.axisImpulse))
        && all(isfinite(s.setup))
        && all(isfinite(s.distanceReference))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

JointSwingState jph_joint_initialize(JointSwingAuthor a, BodyState s1, BodyState s2)
{
    JointSwingState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    s.poseDriveRelative = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        float3 raw = float3(0.0f);
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);

    world_frame = normalize(jph_quat_mul(s2.rotation, a.angularFrame));
    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    s.frame2 = normalize(jph_quat_mul(jph_joint_conjugate(s2.rotation), world_frame));
    s.pointImpulse = float4(0.0f);

    s.angularLimitX = float4(0.0f);
    s.angularLimitY = float4(0.0f);
    s.angularLimitZ = float4(0.0f);
    s.angularLimitImpulse = float4(0.0f);
    s.angularMotorImpulse = float4(0.0f);
    s.angularMotorSetupX = float4(0.0f);
    s.angularMotorSetupY = float4(0.0f);
    s.angularMotorSetupZ = float4(0.0f);

    if (!(all(isfinite(s.poseDriveRelative))
        && all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.frame1))
        && all(isfinite(s.frame2))
        && all(isfinite(s.pointImpulse))
        && all(isfinite(s.angularLimitX))
        && all(isfinite(s.angularLimitY))
        && all(isfinite(s.angularLimitZ))
        && all(isfinite(s.angularLimitImpulse))
        && all(isfinite(s.angularMotorImpulse))
        && all(isfinite(s.angularMotorSetupX))
        && all(isfinite(s.angularMotorSetupY))
        && all(isfinite(s.angularMotorSetupZ))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

JointSixState jph_joint_initialize(JointSixAuthor a, BodyState s1, BodyState s2)
{
    JointSixState s;
    s.flags = uint4(1u, JointStatusSuccess, 0u, 0u);
    s.poseDriveRelative = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        float3 raw = float3(0.0f);
        jph_joint_unscale(offset, a.scale1.xyz, raw, s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);

    world_frame = normalize(jph_quat_mul(s2.rotation, a.angularFrame));
    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    // Six's authored frame is already owner-local and normalized.
    s.frame2 = a.angularFrame;
    s.pointImpulse = float4(0.0f);
    s.angularImpulse = float4(0.0f);
    s.angularLimitX = float4(0.0f);
    s.angularLimitY = float4(0.0f);
    s.angularLimitZ = float4(0.0f);
    s.angularLimitImpulse = float4(0.0f);
    s.angularMotorImpulse = float4(0.0f);
    s.angularMotorSetupX = float4(0.0f);
    s.angularMotorSetupY = float4(0.0f);
    s.angularMotorSetupZ = float4(0.0f);
    s.sixTranslationAxisX = float4(0.0f);
    s.sixTranslationAxisY = float4(0.0f);
    s.sixTranslationAxisZ = float4(0.0f);
    s.sixLimitSetupX = float4(0.0f);
    s.sixLimitSetupY = float4(0.0f);
    s.sixLimitSetupZ = float4(0.0f);
    s.sixMotorSetupX = float4(0.0f);
    s.sixMotorSetupY = float4(0.0f);
    s.sixMotorSetupZ = float4(0.0f);
    s.sixLimitImpulse = float4(0.0f);
    s.sixMotorImpulse = float4(0.0f);

    if (!(all(isfinite(s.poseDriveRelative))
        && all(isfinite(s.anchor1))
        && all(isfinite(s.anchor2))
        && all(isfinite(s.collapsed1))
        && all(isfinite(s.collapsed2))
        && all(isfinite(s.frame1))
        && all(isfinite(s.frame2))
        && all(isfinite(s.pointImpulse))
        && all(isfinite(s.angularImpulse))
        && all(isfinite(s.angularLimitX))
        && all(isfinite(s.angularLimitY))
        && all(isfinite(s.angularLimitZ))
        && all(isfinite(s.angularLimitImpulse))
        && all(isfinite(s.angularMotorImpulse))
        && all(isfinite(s.angularMotorSetupX))
        && all(isfinite(s.angularMotorSetupY))
        && all(isfinite(s.angularMotorSetupZ))
        && all(isfinite(s.sixTranslationAxisX))
        && all(isfinite(s.sixTranslationAxisY))
        && all(isfinite(s.sixTranslationAxisZ))
        && all(isfinite(s.sixLimitSetupX))
        && all(isfinite(s.sixLimitSetupY))
        && all(isfinite(s.sixLimitSetupZ))
        && all(isfinite(s.sixMotorSetupX))
        && all(isfinite(s.sixMotorSetupY))
        && all(isfinite(s.sixMotorSetupZ))
        && all(isfinite(s.sixLimitImpulse))
        && all(isfinite(s.sixMotorImpulse))))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

float3 jph_joint_axis(BodyState s1, JointHingeState s)
{
    return jph_rotate(jph_quat_mul(s1.rotation, s.frame1), float3(0.0f, 0.0f, 1.0f));
}

float4 jph_joint_rotation_error(BodyState s1, BodyState s2, JointFixedState s)
{
    float4 inverse_initial = jph_quat_mul(s.frame2, jph_joint_conjugate(s.frame1));
    return jph_quat_mul(jph_quat_mul(s2.rotation, inverse_initial), jph_joint_conjugate(s1.rotation));
}

float4 jph_joint_rotation_error(BodyState s1, BodyState s2, JointHingeState s)
{
    float4 inverse_initial = jph_quat_mul(s.frame2, jph_joint_conjugate(s.frame1));
    return jph_quat_mul(jph_quat_mul(s2.rotation, inverse_initial), jph_joint_conjugate(s1.rotation));
}

float4 jph_joint_rotation_error(BodyState s1, BodyState s2, JointSliderState s)
{
    float4 inverse_initial = jph_quat_mul(s.frame2, jph_joint_conjugate(s.frame1));
    return jph_quat_mul(jph_quat_mul(s2.rotation, inverse_initial), jph_joint_conjugate(s1.rotation));
}

float jph_joint_angle(BodyState s1, BodyState s2, JointHingeState s)
{
    float4 diff = jph_joint_rotation_error(s1, s2, s);
    // Quat::GetRotationAngle: signed twist, invariant under q -> -q.
    return diff.w == 0.0f ? 3.14159265358979323846f : 2.0f * atan(dot(diff.xyz, jph_joint_axis(s1, s)) / diff.w);
}

float jph_joint_center_angle(float angle)
{
    return angle - 6.28318530717958647692f * floor((angle + 3.14159265358979323846f) / 6.28318530717958647692f);
}

float3 jph_joint_angular_velocity(BodyModel m, BodyState s)
{
    return jph_nonstatic(m) ? s.angularVelocity.xyz : float3(0.0f);
}

void jph_joint_arms(JointFixedAuthor a, JointFixedState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

void jph_joint_arms(JointHingeAuthor a, JointHingeState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

void jph_joint_arms(JointSliderAuthor a, JointSliderState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

void jph_joint_arms(JointDistanceAuthor a, JointDistanceState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

void jph_joint_arms(JointSwingAuthor a, JointSwingState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

void jph_joint_arms(JointSixAuthor a, JointSixState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

float3 jph_joint_inertia_sum(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, float3 value)
{
    return jph_inverse_inertia(m1, s1, value) + jph_inverse_inertia(m2, s2, value);
}

float3 jph_joint_point_response(BodyModel m1, BodyState s1, float3 r1,
                              BodyModel m2, BodyState s2, float3 r2, float3 value)
{
    return jph_inverse_linear_mass(m1, value) + jph_inverse_linear_mass(m2, value)
        - cross(r1, jph_inverse_inertia(m1, s1, cross(r1, value)))
        - cross(r2, jph_inverse_inertia(m2, s2, cross(r2, value)));
}

JphPsdRange jph_joint_reachable(BodyModel m1, BodyState s1, float3 arm1,
    BodyModel m2, BodyState s2, float3 arm2, bool point, uint dimension, float3 row0, float3 row1)
{
    JphPsdRange range = jph_psd_range();
    // No-lock behavior retains the previous strict arithmetic/error path.
    bool restricted = any(m1.mass.yz != float2(0.0f)) || any(m2.mass.yz != float2(0.0f));
    bool full1 = jph_dynamic(m1) && (point ? m1.mass.y == 0.0f : m1.mass.z == 0.0f);
    bool full2 = jph_dynamic(m2) && (point ? m2.mass.y == 0.0f : m2.mass.z == 0.0f);
    if (!restricted || full1 || full2) { range.rank = dimension; return range; }
    // Range-only geometric scaling; physical K and applied impulses below
    // retain the original arms and actual mass/inertia weights.
    if (dimension == 2u)
    {
        float rowScale = max(max(abs(row0.x), max(abs(row0.y), abs(row0.z))),
                             max(abs(row1.x), max(abs(row1.y), abs(row1.z))));
        if (rowScale != 0.0f) { row0 /= rowScale; row1 /= rowScale; }
    }
    JphPsdFactors factors = jph_psd_factors();
    uint structural_bound = 0u;
    for (uint body = 0u; body < 2u; ++body)
    {
        BodyModel model = body == 0u ? m1 : m2;
        float3 arm = body == 0u ? arm1 : arm2;
        if (!jph_dynamic(model)) continue;
        if (point)
        {
            float armScale = max(abs(arm.x), max(abs(arm.y), abs(arm.z)));
            if (armScale != 0.0f) arm /= armScale;
        }
        uint translation_locks = uint(model.mass.y), angular_locks = uint(model.mass.z);
        uint free_translation = countbits((~translation_locks) & 7u);
        uint free_angular = countbits((~angular_locks) & 7u);
        // cross(arm, .) has rank at most two, independent of float roundoff
        // in its three sampled columns. This is geometry, not a small-K test.
        uint angular_bound = point ? (all(arm == float3(0.0f)) ? 0u : min(2u, free_angular)) : free_angular;
        structural_bound += min(dimension, (point ? free_translation : 0u) + angular_bound);
        // For positive raw inertia, range(Pworld * Iinv * Pworld) is the
        // allowed world-axis subspace, independent of the principal frame.
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            float3 coordinate = float3(0.0f); coordinate[axis] = 1.0f;
            if (point && (translation_locks & (1u << axis)) == 0u)
            {
                float3 column = dimension == 3u ? coordinate : float3(dot(row0, coordinate), dot(row1, coordinate), 0.0f);
                jph_psd_add(factors, column);
            }
            if ((angular_locks & (1u << axis)) == 0u)
            {
                float3 column = coordinate;
                if (point) column = cross(column, arm);
                if (dimension == 2u) column = float3(dot(row0, column), dot(row1, column), 0.0f);
                jph_psd_add(factors, column);
            }
        }
    }
    factors.upperBound = min(dimension, structural_bound);
    return jph_psd_finish(factors, dimension);
}

float3 jph_joint_solve3(float3 c0, float3 c1, float3 c2, float3 rhs, JphPsdRange reachable, inout uint status)
{
    if (!reachable.valid || reachable.rank < 3u)
    {
        float3 result;
        if (!jph_psd_solve3(c0, c1, c2, rhs, reachable, result)) status = JointStatusNumericalFailure;
        return result;
    }
    // Symmetric diagonal equilibration keeps unrelated units / large inertia
    // ratios from deciding the determinant threshold. This is a local block.
    float3 diagonal = float3(c0.x, c1.y, c2.z);
    if (any(diagonal <= 0.0f) || !all(isfinite(diagonal)))
    {
        status = JointStatusNumericalFailure;
        return float3(0.0f);
    }
    float3 d = rsqrt(diagonal);
    c0 *= d * d.x;
    c1 *= d * d.y;
    c2 *= d * d.z;
    float3 row0 = cross(c1, c2);
    float3 row1 = cross(c2, c0);
    float3 row2 = cross(c0, c1);
    float det = dot(c0, row0);
    float3 scaled_rhs = d * rhs;
    float3 result = d * float3(dot(row0, scaled_rhs), dot(row1, scaled_rhs), dot(row2, scaled_rhs)) / det;
    if (!(det > 0.0f) || !all(isfinite(result)))
    {
        status = JointStatusNumericalFailure;
        return float3(0.0f);
    }
    return result;
}

float3 jph_joint_solve_rotation(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
                              float3 rhs, inout uint status)
{
    return jph_joint_solve3(jph_joint_inertia_sum(m1, s1, m2, s2, float3(1.0f, 0.0f, 0.0f)),
        jph_joint_inertia_sum(m1, s1, m2, s2, float3(0.0f, 1.0f, 0.0f)),
        jph_joint_inertia_sum(m1, s1, m2, s2, float3(0.0f, 0.0f, 1.0f)), rhs,
        jph_joint_reachable(m1,s1,float3(0),m2,s2,float3(0),false,3u,float3(0),float3(0)),status);
}

float3 jph_joint_solve_point(BodyModel m1, BodyState s1, float3 r1, BodyModel m2, BodyState s2, float3 r2,
                           float3 rhs, inout uint status)
{
    return jph_joint_solve3(jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(1.0f, 0.0f, 0.0f)),
        jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(0.0f, 1.0f, 0.0f)),
        jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(0.0f, 0.0f, 1.0f)), rhs,
        jph_joint_reachable(m1,s1,r1,m2,s2,r2,true,3u,float3(0),float3(0)),status);
}

void jph_joint_apply_angular(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                             float3 impulse, bool position)
{
    float3 a = jph_inverse_inertia(m1, s1, impulse);
    float3 b = jph_inverse_inertia(m2, s2, impulse);
    if (jph_dynamic(m1))
        if (position) { s1.rotation = jph_rotation_step(s1.rotation, -a); jph_project_motion(m1, s1); }
        else s1.angularVelocity.xyz -= a;
    if (jph_dynamic(m2))
        if (position) { s2.rotation = jph_rotation_step(s2.rotation, b); jph_project_motion(m2, s2); }
        else s2.angularVelocity.xyz += b;
}

void jph_joint_apply_point(BodyModel m1, inout BodyState s1, float3 r1,
                          BodyModel m2, inout BodyState s2, float3 r2, float3 impulse, bool position)
{
    float3 a = jph_inverse_inertia(m1, s1, cross(r1, impulse));
    float3 b = jph_inverse_inertia(m2, s2, cross(r2, impulse));
    if (jph_dynamic(m1))
        if (position)
        {
            s1.positionCOM.xyz -= jph_inverse_linear_mass(m1, impulse);
            s1.rotation = jph_rotation_step(s1.rotation, -a);
            jph_project_motion(m1, s1);
        }
        else
        {
            s1.linearVelocity.xyz -= jph_inverse_linear_mass(m1, impulse);
            s1.angularVelocity.xyz -= a;
        }
    if (jph_dynamic(m2))
        if (position)
        {
            s2.positionCOM.xyz += jph_inverse_linear_mass(m2, impulse);
            s2.rotation = jph_rotation_step(s2.rotation, b);
            jph_project_motion(m2, s2);
        }
        else
        {
            s2.linearVelocity.xyz += jph_inverse_linear_mass(m2, impulse);
            s2.angularVelocity.xyz += b;
        }
}

void jph_joint_swing_basis(BodyState s1, BodyState s2, JointHingeState state,
                          out float3 j0, out float3 j1, out float2 error)
{
    float3 a1 = jph_joint_axis(s1, state);
    float3 a2 = jph_rotate(jph_quat_mul(s2.rotation, state.frame2), float3(0.0f, 0.0f, 1.0f));
    float alignment = dot(a1, a2);
    if (alignment <= 1.0e-3f)
    {
        float3 perpendicular = a2 - alignment * a1;
        if (dot(perpendicular, perpendicular) < 1.0e-6f)
            perpendicular = jph_joint_perpendicular(a1);
        a2 = normalize(0.99f * normalize(perpendicular) + 0.01f * a1);
    }
    float3 b2 = jph_joint_perpendicular(a2);
    float3 c2 = cross(a2, b2);
    j0 = cross(b2, a1);
    j1 = cross(c2, a1);
    error = float2(dot(a1, b2), dot(a1, c2));
}

float2 jph_joint_symmetric2(float a, float b, float c, float2 rhs, JphPsdRange reachable, inout uint status)
{
    if (!reachable.valid || reachable.rank < 2u)
    {
        float2 result;
        if (!jph_psd_solve2(a,b,c,rhs,reachable,result)) status = JointStatusNumericalFailure;
        return result;
    }
    float2 d = rsqrt(float2(a, c));
    float off = b * d.x * d.y;
    float det = 1.0f - off * off;
    float2 scaled_rhs = rhs * d;
    float2 result = d * float2(scaled_rhs.x - off * scaled_rhs.y, scaled_rhs.y - off * scaled_rhs.x) / det;
    if (!(a > 0.0f && c > 0.0f && det > 0.0f) || !all(isfinite(result)))
    {
        status = JointStatusNumericalFailure;
        return float2(0.0f);
    }
    return result;
}

float2 jph_joint_solve2(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
                       float3 j0, float3 j1, float2 rhs, inout uint status)
{
    float3 i0 = jph_joint_inertia_sum(m1, s1, m2, s2, j0);
    float3 i1 = jph_joint_inertia_sum(m1, s1, m2, s2, j1);
    return jph_joint_symmetric2(dot(j0, i0), dot(j0, i1), dot(j1, i1), rhs,
        jph_joint_reachable(m1,s1,float3(0),m2,s2,float3(0),false,2u,j0,j1),status);
}

// Slider rows use the same body writer and phase schedule as Fixed/Hinge.
// Mathematical source: SliderConstraint / DualAxisConstraintPart / AxisConstraintPart.
void jph_slider_basis(BodyState s1, JointSliderState state, out float3 axis, out float3 n0, out float3 n1)
{
    float4 frame = jph_quat_mul(s1.rotation, state.frame1);
    axis = jph_rotate(frame, float3(1.0f, 0.0f, 0.0f));
    n0 = jph_rotate(frame, float3(0.0f, 1.0f, 0.0f));
    n1 = jph_rotate(frame, float3(0.0f, 0.0f, 1.0f));
}

void jph_slider_limit(JointSliderAuthor a, float position, out float side, out float error)
{
    side = 0.0f;
    error = 0.0f;
    if ((a.bodies.w & JointLimitsEnabled) == 0u) return;
    if (position <= a.limits.x)
    {
        side = -1.0f;
        error = position - a.limits.x;
    }
    else if (position >= a.limits.y)
    {
        side = 1.0f;
        error = position - a.limits.y;
    }
}

float2 jph_slider_transverse(BodyModel m1, BodyState s1, float3 arm1,
    BodyModel m2, BodyState s2, float3 r2, float3 n0, float3 n1, float2 rhs, inout uint status)
{
    float3 k0 = jph_joint_point_response(m1, s1, arm1, m2, s2, r2, n0);
    float3 k1 = jph_joint_point_response(m1, s1, arm1, m2, s2, r2, n1);
    return jph_joint_symmetric2(dot(n0, k0), dot(n0, k1), dot(n1, k1), rhs,
        jph_joint_reachable(m1,s1,arm1,m2,s2,r2,true,2u,n0,n1),status);
}

void jph_slider_solve_velocity(JointSliderAuthor a, inout JointSliderState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 arm1 = r1 + u; // The rotating reference axis contributes u to body 1's Jacobian.
    float3 axis, n0, n1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float3 relative = jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, r2);
    float2 transverse = jph_slider_transverse(m1, s1, arm1, m2, s2, r2, n0, n1,
        float2(dot(n0, relative), dot(n1, relative)), state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    state.pointImpulse.xy += transverse;
    jph_joint_apply_point(m1, s1, arm1, m2, s2, r2, n0 * transverse.x + n1 * transverse.y, false);
    float3 angular = jph_joint_solve_rotation(m1, s1, m2, s2,
        jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2), state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    state.angularImpulse.xyz += angular;
    jph_joint_apply_angular(m1, s1, m2, s2, angular, false);
    if (state.setup.w != 0.0f)
    {
        float k = dot(axis, jph_joint_point_response(m1, s1, arm1, m2, s2, r2, axis));
        if (k == 0.0f) { state.axisImpulse.x = 0.0f; return; }
        if (!(k > 0.0f) || !isfinite(k))
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
        float old = state.axisImpulse.x;
        relative = jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, r2);
        float next = old + dot(axis, relative) / k;
        if (a.limits.x != a.limits.y)
            next = state.setup.w < 0.0f ? max(0.0f, next) : min(0.0f, next);
        state.axisImpulse.x = next;
        jph_joint_apply_point(m1, s1, arm1, m2, s2, r2, axis * (next - old), false);
    }
}

void jph_slider_solve_position(JointSliderAuthor a, inout JointSliderState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 axis, n0, n1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float2 transverse = jph_slider_transverse(m1, s1, r1 + u, m2, s2, r2, n0, n1,
        -baumgarte * float2(dot(n0, u), dot(n1, u)), state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2, n0 * transverse.x + n1 * transverse.y, true);
    float4 diff = jph_joint_rotation_error(s1, s2, state);
    float3 error = 2.0f * (diff.w < 0.0f ? -diff.xyz : diff.xyz);
    float3 angular = jph_joint_solve_rotation(m1, s1, m2, s2, -baumgarte * error, state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    jph_joint_apply_angular(m1, s1, m2, s2, angular, true);
    // Rotation correction changes both material points and the reference basis.
    jph_joint_arms(a, state, s1, s2, r1, r2);
    u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float side, limit_error;
    jph_slider_limit(a, dot(axis, u), side, limit_error);
    if (side != 0.0f && limit_error != 0.0f)
    {
        float k = dot(axis, jph_joint_point_response(m1, s1, r1 + u, m2, s2, r2, axis));
        if (k == 0.0f) { state.axisImpulse.x = 0.0f; return; }
        if (!(k > 0.0f) || !isfinite(k))
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
        jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2, axis * (-baumgarte * limit_error / k), true);
    }
}

// DistanceConstraint + AxisConstraintPart, StiffnessAndDamping limit mode.
void jph_distance_limit(JointDistanceAuthor a, JointDistanceState state, float coordinate,
    out float side, out float error)
{
    float minimum = (a.bodies.w & JointAutoDistance) != 0u ? state.distanceReference.w : a.limits.x;
    float maximum = (a.bodies.w & JointAutoDistance) != 0u ? state.distanceReference.w : a.limits.y;
    side = 0.0f;
    error = 0.0f;
    if (minimum == maximum)
    {
        side = 2.0f; // Bilateral, including equality at zero length.
        error = coordinate - minimum;
    }
    else if (coordinate <= minimum)
    {
        side = -1.0f;
        error = coordinate - minimum;
    }
    else if (coordinate >= maximum)
    {
        side = 1.0f;
        error = coordinate - maximum;
    }
}

float jph_distance_geometry(JointDistanceAuthor a, inout JointDistanceState state, BodyState s1, BodyState s2,
    out float3 arm1, out float3 r2)
{
    float3 r1;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float coordinate = length(u);
    if (coordinate > 0.0f) state.distanceReference.xyz = u / coordinate;
    // Preserve Jolt's cached dot(delta, normal), not a new pose evaluation
    // during the later position-error test (or an assumed bit-identical sqrt).
    state.setup.x = dot(u, state.distanceReference.xyz);
    arm1 = r1 + u;
    return coordinate;
}

float jph_distance_inverse_mass(BodyModel m1, BodyState s1, float3 arm1,
    BodyModel m2, BodyState s2, float3 r2, inout JointDistanceState state)
{
    float3 n = state.distanceReference.xyz;
    float k = dot(n, jph_joint_point_response(m1, s1, arm1, m2, s2, r2, n));
    if (k < 0.0f || !isfinite(k)) state.flags.y = JointStatusNumericalFailure;
    return k;
}

void jph_distance_setup(JointDistanceAuthor a, inout JointDistanceState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, float dt)
{
    float3 arm1, r2;
    float coordinate = jph_distance_geometry(a, state, s1, s2, arm1, r2);
    float side, error;
    jph_distance_limit(a, state, coordinate, side, error);
    state.setup = float4(state.setup.x, 0.0f, 0.0f, side);

    state.axisImpulse.yzw = float3(0.0f);

    // An interior row is inactive before inverse-mass evaluation in Jolt.
    float k = side == 0.0f ? 0.0f : jph_distance_inverse_mass(m1, s1, arm1, m2, s2, r2, state);
    if (side == 0.0f || k == 0.0f)
    {
        state.setup.w = 0.0f;
        state.axisImpulse.x = 0.0f;
    }
    else if (a.linearSpring.x > 0.0f)
    {
        // HasStiffness, not HasStiffnessOrDamping: damping-only remains hard.
        state.setup.y = 1.0f / (dt * (a.linearSpring.y + dt * a.linearSpring.x));
        state.setup.z = dt * a.linearSpring.x * state.setup.y * error;
    }
    if (!all(isfinite(state.setup)) || !all(isfinite(state.distanceReference)))
        state.flags.y = JointStatusNumericalFailure;
}

void jph_distance_solve_velocity(JointDistanceAuthor a, inout JointDistanceState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.setup.w == 0.0f) return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 arm1 = s2.positionCOM.xyz - s1.positionCOM.xyz + r2;
    float k = jph_distance_inverse_mass(m1, s1, arm1, m2, s2, r2, state);
    if (k == 0.0f) { state.axisImpulse.x = 0.0f; state.setup.w = 0.0f; return; }
    float denominator = k + state.setup.y;
    if (state.flags.y != JointStatusSuccess || !(denominator > 0.0f) || !isfinite(denominator))
    {
        state.flags.y = JointStatusNumericalFailure;
        return;
    }
    float3 n = state.distanceReference.xyz;
    float jv = dot(n, jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, r2));
    float old = state.axisImpulse.x;
    float next = old + (jv - state.setup.y * old - state.setup.z) / denominator;
    if (state.setup.w == -1.0f) next = max(0.0f, next);
    else if (state.setup.w == 1.0f) next = min(0.0f, next);
    state.axisImpulse.x = next;
    jph_joint_apply_point(m1, s1, arm1, m2, s2, r2, n * (next - old), false);
}

void jph_distance_solve_position(JointDistanceAuthor a, inout JointDistanceState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (a.linearSpring.x > 0.0f) return;
    // Jolt checks the cached points before recalculating properties. The last
    // projected distance is their length, including the retained zero normal.
    float side, error;
    jph_distance_limit(a, state, state.setup.x, side, error);
    if (error == 0.0f) return;
    float3 arm1, r2;
    float coordinate = jph_distance_geometry(a, state, s1, s2, arm1, r2);
    float current_error;
    jph_distance_limit(a, state, coordinate, state.setup.w, current_error);
    if (state.setup.w == 0.0f)
    {
        state.axisImpulse.x = 0.0f;
        return;
    }
    float k = jph_distance_inverse_mass(m1, s1, arm1, m2, s2, r2, state);
    if (state.flags.y != JointStatusSuccess || k == 0.0f)
    {
        state.axisImpulse.x = 0.0f;
        return;
    }
    jph_joint_apply_point(m1, s1, arm1, m2, s2, r2,
        state.distanceReference.xyz * (-baumgarte * error / k), true);
}

void jph_joint_limit(JointHingeAuthor a, float angle, out float side, out float error)
{
    side = 0.0f;
    error = 0.0f;
    bool has_limits = (a.bodies.w & JointLimitsEnabled) != 0u
        && (a.limits.x > -3.14159265358979323846f || a.limits.y < 3.14159265358979323846f);
    if (has_limits && (angle <= a.limits.x || angle >= a.limits.y))
    {
        float lower = jph_joint_center_angle(angle - a.limits.x);
        float upper = jph_joint_center_angle(angle - a.limits.y);
        bool use_lower = abs(lower) < abs(upper);
        side = use_lower ? -1.0f : 1.0f;
        error = use_lower ? lower : upper;
    }
}

bool jph_joint_drive_active(JointHingeAuthor a)
{
    return ((a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f))
        || any(a.passive.xyz > 0.0f);
}

void jph_joint_clear_impulses(inout JointFixedState state)
{
    state.pointImpulse = float4(0.0f);
    state.angularImpulse = float4(0.0f);
}

void jph_joint_clear_impulses(inout JointHingeState state)
{
    state.pointImpulse = float4(0.0f);
    state.angularImpulse = float4(0.0f);
    state.axisImpulse = float4(0.0f);
    state.rotor.y = 0.0f;
}

void jph_joint_clear_impulses(inout JointSliderState state)
{
    state.pointImpulse = float4(0.0f);
    state.angularImpulse = float4(0.0f);
    state.axisImpulse = float4(0.0f);
}

void jph_joint_clear_impulses(inout JointDistanceState state)
{
    state.axisImpulse = float4(0.0f);
}

void jph_joint_clear_impulses(inout JointSwingState state)
{
    state.pointImpulse = float4(0.0f);
    state.angularLimitImpulse = float4(0.0f);
    state.angularMotorImpulse = float4(0.0f);
}

void jph_joint_clear_impulses(inout JointSixState state)
{
    state.pointImpulse = state.angularImpulse = float4(0.0f);
    state.angularLimitImpulse = state.angularMotorImpulse = float4(0.0f);
    state.sixLimitImpulse = state.sixMotorImpulse = float4(0.0f);
}

#include "rigid_swing_twist.slang"

float3 jph_joint_project_range(JphPsdRange range, float3 impulse, uint dimension, inout uint status)
{
    if (!range.valid) { status = JointStatusNumericalFailure; return float3(0.0f); }
    if (range.rank == dimension) return impulse;
    float3 result = float3(0.0f);
    for (uint i = 0u; i < range.rank; ++i) result += range.axis[i] * dot(range.axis[i], impulse);
    return result;
}

void jph_joint_project_warm(JointFixedAuthor a, inout JointFixedState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2)
{
    if (all(m1.mass.yz == float2(0.0f)) && all(m2.mass.yz == float2(0.0f))) return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
     // Scalar setup already clears K=0.
    state.pointImpulse.xyz = jph_joint_project_range(
        jph_joint_reachable(m1,s1,r1,m2,s2,r2,true,3u,float3(0),float3(0)),state.pointImpulse.xyz,3u, state.flags.y);
    state.angularImpulse.xyz = jph_joint_project_range(
            jph_joint_reachable(m1,s1,float3(0),m2,s2,float3(0),false,3u,float3(0),float3(0)),
            state.angularImpulse.xyz,3u, state.flags.y);
}

void jph_joint_project_warm(JointHingeAuthor a, inout JointHingeState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2)
{
    if (all(m1.mass.yz == float2(0.0f)) && all(m2.mass.yz == float2(0.0f))) return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
     // Scalar setup already clears K=0.
    state.pointImpulse.xyz = jph_joint_project_range(
        jph_joint_reachable(m1,s1,r1,m2,s2,r2,true,3u,float3(0),float3(0)),state.pointImpulse.xyz,3u, state.flags.y);
    {
        float3 j0,j1; float2 error;
        jph_joint_swing_basis(s1,s2,state,j0,j1,error);
        state.angularImpulse.xy = jph_joint_project_range(
            jph_joint_reachable(m1,s1,float3(0),m2,s2,float3(0),false,2u,j0,j1),
            float3(state.angularImpulse.xy,0.0f),2u, state.flags.y).xy;
        float3 axis = jph_joint_axis(s1,state);
        if (dot(axis,jph_joint_inertia_sum(m1,s1,m2,s2,axis)) == 0.0f)
        { state.axisImpulse = float4(0.0f); state.setup.w = 0.0f; state.rotor.y = 0.0f; }
    }
}

void jph_joint_project_warm(JointSliderAuthor a, inout JointSliderState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2)
{
    if (all(m1.mass.yz == float2(0.0f)) && all(m2.mass.yz == float2(0.0f))) return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
     // Scalar setup already clears K=0.
    {
        float3 axis, n0, n1;
        jph_slider_basis(s1, state, axis, n0, n1);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        state.pointImpulse.xy = jph_joint_project_range(
            jph_joint_reachable(m1,s1,r1+u,m2,s2,r2,true,2u,n0,n1),
            float3(state.pointImpulse.xy,0.0f),2u, state.flags.y).xy;
        if (dot(axis,jph_joint_point_response(m1,s1,r1+u,m2,s2,r2,axis)) == 0.0f)
        { state.axisImpulse.x = 0.0f; state.setup.w = 0.0f; }
    }
    state.angularImpulse.xyz = jph_joint_project_range(
            jph_joint_reachable(m1,s1,float3(0),m2,s2,float3(0),false,3u,float3(0),float3(0)),
            state.angularImpulse.xyz,3u, state.flags.y);
}

void jph_joint_project_warm(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2)
{
    if (all(m1.mass.yz == float2(0.0f)) && all(m2.mass.yz == float2(0.0f))) return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
     // Scalar setup already clears K=0.
    state.pointImpulse.xyz = jph_joint_project_range(
        jph_joint_reachable(m1,s1,r1,m2,s2,r2,true,3u,float3(0),float3(0)),state.pointImpulse.xyz,3u, state.flags.y);

}

void jph_joint_setup(JointFixedAuthor a, inout JointFixedState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float previous_dt = asfloat(state.flags.z);
    float ratio = previous_dt > 0.0f ? dt / previous_dt : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }

    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    state.pointImpulse *= ratio;
    state.angularImpulse *= ratio;

    jph_joint_project_warm(a,state,m1,s1,m2,s2);
}

void jph_joint_setup(JointHingeAuthor a, inout JointHingeState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if ((state.flags.w & JointPoseDriveActive) != 0u)
    {
        float4 target = normalize(jph_quat_mul(jph_quat_mul(jph_joint_conjugate(state.frame1), state.poseDriveRelative), state.frame2));
        a.motor.x = target.w == 0.0f ? 3.14159265358979323846f : 2.0f * atan(target.z / target.w);
    }

    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float previous_dt = asfloat(state.flags.z);
    float ratio = previous_dt > 0.0f ? dt / previous_dt : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }
    {
        float3 axis = jph_joint_axis(s1, state);
        state.rotor.z = a.passive.z > 0.0f ? state.rotor.x - state.rotor.y / a.passive.z : 0.0f;
        state.rotor.x = state.rotor.z;
        state.setup.x = jph_joint_angle(s1, s2, state);
        float limit_error;
        jph_joint_limit(a, state.setup.x, state.setup.w, limit_error);
        if (state.setup.w == 0.0f)
            state.axisImpulse.x = 0.0f;
        state.setup.yz = float2(0.0f);
        if ((a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f))
        {
            float target = (a.bodies.w & JointLimitsEnabled) != 0u ? clamp(a.motor.x, a.limits.x, a.limits.y) : a.motor.x;
            state.setup.y = 1.0f / (dt * (a.motor.z + dt * a.motor.y));
            state.setup.z = dt * a.motor.y * state.setup.y * jph_joint_center_angle(state.setup.x - target);
        }
        if (!jph_joint_drive_active(a))
        {
            state.axisImpulse.yzw = float3(0.0f);
            state.rotor.xy = float2(0.0f);
        }
    }

    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    state.pointImpulse *= ratio;
    state.angularImpulse *= ratio;
    state.axisImpulse *= ratio;
    state.rotor.y *= ratio;

    jph_joint_project_warm(a,state,m1,s1,m2,s2);
}

void jph_joint_setup(JointSliderAuthor a, inout JointSliderState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float previous_dt = asfloat(state.flags.z);
    float ratio = previous_dt > 0.0f ? dt / previous_dt : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }

    {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        float3 axis, n0, n1;
        jph_slider_basis(s1, state, axis, n0, n1);
        state.setup = float4(dot(axis, u), 0.0f, 0.0f, 0.0f);
        float error;
        jph_slider_limit(a, state.setup.x, state.setup.w, error);
        if (state.setup.w == 0.0f) state.axisImpulse.x = 0.0f;
        state.pointImpulse.zw = float2(0.0f);
        state.axisImpulse.yzw = float3(0.0f);

    }

    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    state.pointImpulse *= ratio;
    state.angularImpulse *= ratio;
    state.axisImpulse *= ratio;

    jph_joint_project_warm(a,state,m1,s1,m2,s2);
}

void jph_joint_setup(JointDistanceAuthor a, inout JointDistanceState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float previous_dt = asfloat(state.flags.z);
    float ratio = previous_dt > 0.0f ? dt / previous_dt : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }

    jph_distance_setup(a, state, m1, s1, m2, s2, dt);

    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }

    state.axisImpulse *= ratio;

}

void jph_joint_setup(JointSwingAuthor a, inout JointSwingState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float previous_dt = asfloat(state.flags.z);
    float ratio = previous_dt > 0.0f ? dt / previous_dt : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }

    jph_swing_setup(a, state, m1, s1, m2, s2, dt);
    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    state.pointImpulse *= ratio;

    {
        state.angularLimitImpulse *= ratio;
        state.angularMotorImpulse *= ratio;
    }
    jph_joint_project_warm(a,state,m1,s1,m2,s2);
}

void jph_joint_warm_start(JointFixedAuthor a, JointFixedState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);

    {
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularImpulse.xyz, false);
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, state.pointImpulse.xyz, false);
    }
}

void jph_joint_warm_start(JointHingeAuthor a, JointHingeState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);

    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float3 axis = jph_joint_axis(s1, state);
        jph_joint_apply_angular(m1, s1, m2, s2, axis * state.axisImpulse.w, false);
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, state.pointImpulse.xyz, false);
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * state.angularImpulse.x + j1 * state.angularImpulse.y, false);
        jph_joint_apply_angular(m1, s1, m2, s2, axis * state.axisImpulse.x, false);
    }
}

void jph_joint_warm_start(JointSliderAuthor a, JointSliderState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    {
        float3 axis, n0, n1;
        jph_slider_basis(s1, state, axis, n0, n1);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2,
            n0 * state.pointImpulse.x + n1 * state.pointImpulse.y + axis * state.axisImpulse.x, false);
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularImpulse.xyz, false);
        return;
    }
}

void jph_joint_warm_start(JointDistanceAuthor a, JointDistanceState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);

    {
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2,
            state.distanceReference.xyz * state.axisImpulse.x, false);
        return;
    }
}

void jph_joint_warm_start(JointSwingAuthor a, JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    {
        jph_swing_warm(a, state, m1, s1, m2, s2);
        return;
    }
}

struct JphJointDriveImpulse { float motor, passive, total, rotor, velocity; };

void jph_joint_apply_drive_velocity(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
    float3 axis, float3 response1, float3 response2, float k1, float k2, float velocity, inout uint status)
{
    float k = k1 + k2;
    if (!(k > 0.0f) || k1 < 0.0f || k2 < 0.0f || !isfinite(k)
        || !isfinite(velocity) || !all(isfinite(response1)) || !all(isfinite(response2)))
    {
        status = JointStatusNumericalFailure;
        return;
    }
    float3 w1 = jph_joint_angular_velocity(m1, s1);
    float3 w2 = jph_joint_angular_velocity(m2, s2);
    float u = dot(axis, w1), v = dot(axis, w2);
    float weight1 = k1 / k, weight2 = k2 / k;
    // Equal and opposite axial impulses leave this common projection intact.
    // Reconstruct along inverse-inertia response, including its off-axis terms.
    float common = fma(weight2, u, weight1 * v);
    float next_u = fma(-weight1, velocity, common);
    float next_v = fma(weight2, velocity, common);
    float3 next1 = s1.angularVelocity.xyz, next2 = s2.angularVelocity.xyz;
    if (jph_dynamic(m1) && k1 > 0.0f)
    {
        float3 direction = response1 / k1;
        float3 residual = fma(-direction, float3(u), w1);
        next1 = fma(direction, float3(next_u), residual);
    }
    if (jph_dynamic(m2) && k2 > 0.0f)
    {
        float3 direction = response2 / k2;
        float3 residual = fma(-direction, float3(v), w2);
        next2 = fma(direction, float3(next_v), residual);
    }
    if (!isfinite(common) || !isfinite(next_u) || !isfinite(next_v)
        || !all(isfinite(next1)) || !all(isfinite(next2)))
    {
        status = JointStatusNumericalFailure;
        return;
    }
    if (jph_dynamic(m1) && k1 > 0.0f) s1.angularVelocity.xyz = next1;
    if (jph_dynamic(m2) && k2 > 0.0f) s2.angularVelocity.xyz = next2;
}

JphJointDriveImpulse jph_joint_drive(float free_velocity, float k, float gamma, float bias,
    float min_motor, float max_motor, float damping, float friction, float armature, float old_speed,
    inout uint status)
{
    // ConstraintPart/DriveImpulse.h. Keep the stable net-impulse expression:
    // independent co-linear motor/passive rows undo one another on light links.
    // Evaluate net velocity independently of the cancelling free + K * lambda.
    JphJointDriveImpulse result;
    result.motor = result.passive = result.total = result.rotor = result.velocity = 0.0f;
    float shifted = free_velocity + k * armature * old_speed;
    float motor;
    if (gamma > 0.0f) motor = clamp(-bias / gamma, min_motor, max_motor);
    else if (bias < 0.0f) motor = max_motor;
    else if (bias > 0.0f) motor = min_motor;
    else motor = clamp(-shifted / k, min_motor, max_motor);
    float at_rest = shifted + k * motor;
    float friction_bound = k * friction;
    float signed_friction = at_rest > 0.0f ? friction : -friction;
    float b = 1.0f + k * (damping + armature);
    float motor_rhs = shifted - k * signed_friction + b * bias;
    float motor_denominator = b * gamma + k;
    if (!(b > 0.0f) || !(motor_denominator > 0.0f)
        || !all(isfinite(float4(min_motor, max_motor, shifted, at_rest)))
        || !all(isfinite(float4(friction_bound, b, motor_rhs, motor_denominator))))
    {
        status = JointStatusNumericalFailure;
        return result;
    }
    if (abs(at_rest) <= friction_bound)
    {
        result.total = -free_velocity / k;
        result.rotor = armature * old_speed;
        result.motor = motor;
        result.passive = result.total - motor - result.rotor;
        result.velocity = 0.0f;
        return result;
    }
    motor = clamp(-motor_rhs / motor_denominator, min_motor, max_motor);
    result.total = (motor - signed_friction - damping * free_velocity + armature * (old_speed - free_velocity)) / b;
    float velocity = (free_velocity + k * (motor - signed_friction + armature * old_speed)) / b;
    result.velocity = velocity;
    result.motor = motor;
    result.passive = -damping * velocity - signed_friction;
    result.rotor = (armature / b) * (old_speed - free_velocity + k * (damping * old_speed - motor + signed_friction));
    return result;
}

void jph_joint_solve_velocity(JointFixedAuthor a, inout JointFixedState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        float3 lambda = jph_joint_solve_rotation(m1, s1, m2, s2,
            jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2), state.flags.y);
        state.angularImpulse.xyz += lambda;
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, false);
    }
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2,
        jph_point_velocity(m1, s1, r1) - jph_point_velocity(m2, s2, r2), state.flags.y);
    state.pointImpulse.xyz += point_lambda;
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, false);

}

void jph_joint_solve_velocity(JointHingeAuthor a, inout JointHingeState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    if (jph_joint_drive_active(a))
    {
        float3 axis = jph_joint_axis(s1, state);
        float3 response1 = jph_inverse_inertia(m1, s1, axis);
        float3 response2 = jph_inverse_inertia(m2, s2, axis);
        float k1 = dot(axis, response1), k2 = dot(axis, response2);
        float k = k1 + k2;
        if (k > 0.0f && isfinite(k) && k1 >= 0.0f && k2 >= 0.0f)
        {
            float free_velocity = dot(axis, jph_joint_angular_velocity(m2, s2) - jph_joint_angular_velocity(m1, s1)) - k * state.axisImpulse.w;
            bool driven = (a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f);
            float torque = driven ? dt * a.motor.w : 0.0f;
            JphJointDriveImpulse solved = jph_joint_drive(free_velocity, k, state.setup.y, state.setup.z,
                -torque, torque, dt * a.passive.x, dt * a.passive.y, a.passive.z, state.rotor.z, state.flags.y);
            if (state.flags.y != JointStatusSuccess) return;
            if (!isfinite(solved.motor) || !isfinite(solved.passive) || !isfinite(solved.total)
                || !isfinite(solved.rotor) || !isfinite(solved.velocity))
            {
                state.flags.y = JointStatusNumericalFailure;
                return;
            }
            state.axisImpulse.yzw = float3(solved.motor, solved.passive, solved.total);
            state.rotor.y = solved.rotor;
            jph_joint_apply_drive_velocity(m1, s1, m2, s2, axis, response1, response2, k1, k2,
                solved.velocity, state.flags.y);
            if (state.flags.y != JointStatusSuccess) return;
        }
        else if (k == 0.0f && k1 == 0.0f && k2 == 0.0f)
        {
            state.axisImpulse.yzw = float3(0.0f);
            state.rotor.y = 0.0f;
        }
        else
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
    }
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2,
        jph_point_velocity(m1, s1, r1) - jph_point_velocity(m2, s2, r2), state.flags.y);
    state.pointImpulse.xyz += point_lambda;
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, false);
    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float3 dw = jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2);
        float2 lambda = jph_joint_solve2(m1, s1, m2, s2, j0, j1, float2(dot(j0, dw), dot(j1, dw)), state.flags.y);
        state.angularImpulse.xy += lambda;
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * lambda.x + j1 * lambda.y, false);
        if (state.setup.w != 0.0f)
        {
            float3 axis = jph_joint_axis(s1, state);
            float k = dot(axis, jph_joint_inertia_sum(m1, s1, m2, s2, axis));
            if (k > 0.0f)
            {
                float old_lambda = state.axisImpulse.x;
                float next_lambda = old_lambda + dot(axis, jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2)) / k;
                if (a.limits.x != a.limits.y)
                    next_lambda = state.setup.w < 0.0f ? max(0.0f, next_lambda) : min(0.0f, next_lambda);
                state.axisImpulse.x = next_lambda;
                jph_joint_apply_angular(m1, s1, m2, s2, axis * (next_lambda - old_lambda), false);
            }
        }
    }
}

void jph_joint_solve_velocity(JointSliderAuthor a, inout JointSliderState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        jph_slider_solve_velocity(a, state, m1, s1, m2, s2);
        return;
    }

}

void jph_joint_solve_velocity(JointDistanceAuthor a, inout JointDistanceState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        jph_distance_solve_velocity(a, state, m1, s1, m2, s2);
        return;
    }

}

void jph_joint_solve_velocity(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    {
        jph_swing_velocity(a, state, m1, s1, m2, s2);
        return;
    }

}

void jph_joint_solve_position(JointFixedAuthor a, inout JointFixedState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        float4 diff = jph_joint_rotation_error(s1, s2, state);
        float3 error = 2.0f * (diff.w < 0.0f ? -diff.xyz : diff.xyz);
        float3 lambda = jph_joint_solve_rotation(m1, s1, m2, s2, -baumgarte * error, state.flags.y);
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, true);
    }
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 separation = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2, -baumgarte * separation, state.flags.y);
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, true);

}

void jph_joint_solve_position(JointHingeAuthor a, inout JointHingeState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 separation = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2, -baumgarte * separation, state.flags.y);
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, true);
    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float2 lambda = jph_joint_solve2(m1, s1, m2, s2, j0, j1, -baumgarte * error, state.flags.y);
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * lambda.x + j1 * lambda.y, true);
        float side, angle_error;
        jph_joint_limit(a, jph_joint_angle(s1, s2, state), side, angle_error);
        if (side != 0.0f)
        {
            float3 axis = jph_joint_axis(s1, state);
            float k = dot(axis, jph_joint_inertia_sum(m1, s1, m2, s2, axis));
            if (k > 0.0f)
                jph_joint_apply_angular(m1, s1, m2, s2, axis * (-baumgarte * angle_error / k), true);
        }
    }
}

void jph_joint_solve_position(JointSliderAuthor a, inout JointSliderState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        jph_slider_solve_position(a, state, m1, s1, m2, s2, baumgarte);
        return;
    }

}

void jph_joint_solve_position(JointDistanceAuthor a, inout JointDistanceState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;

    {
        jph_distance_solve_position(a, state, m1, s1, m2, s2, baumgarte);
        return;
    }

}

void jph_joint_solve_position(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    {
        jph_swing_position(a, state, m1, s1, m2, s2, baumgarte);
        return;
    }

}

bool jph_joint_finite(BodyState state)
{
    return all(isfinite(state.positionCOM)) && all(isfinite(state.rotation))
        && all(isfinite(state.linearVelocity)) && all(isfinite(state.angularVelocity));
}

bool jph_joint_state_finite(JointFixedState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.frame1))
        && all(isfinite(state.frame2))
        && all(isfinite(state.pointImpulse))
        && all(isfinite(state.angularImpulse));
}

bool jph_joint_state_finite(JointHingeState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.poseDriveRelative))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.frame1))
        && all(isfinite(state.frame2))
        && all(isfinite(state.pointImpulse))
        && all(isfinite(state.angularImpulse))
        && all(isfinite(state.axisImpulse))
        && all(isfinite(state.rotor))
        && all(isfinite(state.setup));
}

bool jph_joint_state_finite(JointSliderState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.frame1))
        && all(isfinite(state.frame2))
        && all(isfinite(state.pointImpulse))
        && all(isfinite(state.angularImpulse))
        && all(isfinite(state.axisImpulse))
        && all(isfinite(state.setup));
}

bool jph_joint_state_finite(JointDistanceState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.axisImpulse))
        && all(isfinite(state.setup))
        && all(isfinite(state.distanceReference));
}

bool jph_joint_state_finite(JointSwingState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.poseDriveRelative))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.frame1))
        && all(isfinite(state.frame2))
        && all(isfinite(state.pointImpulse))
        && all(isfinite(state.angularLimitX))
        && all(isfinite(state.angularLimitY))
        && all(isfinite(state.angularLimitZ))
        && all(isfinite(state.angularLimitImpulse))
        && all(isfinite(state.angularMotorImpulse))
        && all(isfinite(state.angularMotorSetupX))
        && all(isfinite(state.angularMotorSetupY))
        && all(isfinite(state.angularMotorSetupZ));
}

bool jph_joint_state_finite(JointSixState state)
{
    return isfinite(asfloat(state.flags.z))
        && all(isfinite(state.poseDriveRelative))
        && all(isfinite(state.anchor1))
        && all(isfinite(state.anchor2))
        && all(isfinite(state.collapsed1))
        && all(isfinite(state.collapsed2))
        && all(isfinite(state.frame1))
        && all(isfinite(state.frame2))
        && all(isfinite(state.pointImpulse))
        && all(isfinite(state.angularImpulse))
        && all(isfinite(state.angularLimitX))
        && all(isfinite(state.angularLimitY))
        && all(isfinite(state.angularLimitZ))
        && all(isfinite(state.angularLimitImpulse))
        && all(isfinite(state.angularMotorImpulse))
        && all(isfinite(state.angularMotorSetupX))
        && all(isfinite(state.angularMotorSetupY))
        && all(isfinite(state.angularMotorSetupZ))
        && all(isfinite(state.sixTranslationAxisX))
        && all(isfinite(state.sixTranslationAxisY))
        && all(isfinite(state.sixTranslationAxisZ))
        && all(isfinite(state.sixLimitSetupX))
        && all(isfinite(state.sixLimitSetupY))
        && all(isfinite(state.sixLimitSetupZ))
        && all(isfinite(state.sixMotorSetupX))
        && all(isfinite(state.sixMotorSetupY))
        && all(isfinite(state.sixMotorSetupZ))
        && all(isfinite(state.sixLimitImpulse))
        && all(isfinite(state.sixMotorImpulse));
}

#include "rigid_six_dof.slang"

bool jph_joint_suppresses_pair(WorldRange world, uint body1, uint body2)
{
    for (uint local = 0u; local < world.leaves.w; ++local)
    {
        JointDirectory a = jph_joint_load_directory(world.leaves.z + local);
        if ((a.bodies.w & (JointEnabled | JointCollisionEnabled)) == JointEnabled &&
            ((a.bodies.x == body1 && a.bodies.y == body2) || (a.bodies.x == body2 && a.bodies.y == body1)))
            return true;
    }
    return false;
}

#endif // JPH_GPU_RIGID_JOINTS_SLANG
