// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_RIGID_SWING_TWIST_SLANG
#define JPH_GPU_RIGID_SWING_TWIST_SLANG

// SwingTwistConstraint/ConstraintPart law, using the same island owner.
// rigid_joints includes this after its shared point/rotation helpers.
static const uint SwTwistLocked = 1u, SwYLocked = 2u, SwZLocked = 4u;
static const uint SwTwistFree = 8u, SwYFree = 16u, SwZFree = 32u;
static const uint SwClampTwistMin = 1u, SwClampTwistMax = 2u;
static const uint SwClampYMin = 4u, SwClampYMax = 8u, SwClampZMin = 16u, SwClampZMax = 32u;

struct JphSwingLimits
{
    uint flags;
    float2 twistSin, twistCos, ySin, yCos, zSin, zCos;
    float2 halfSwing;
    float2 halfMinimum, halfMaximum;
};

JphSwingLimits jph_swing_limits(JointSwingAuthor a)
{
    JphSwingLimits c;
    c.flags = 0u;
    c.twistSin = sin(0.5f * a.angularLimits.xy);
    c.twistCos = cos(0.5f * a.angularLimits.xy);
    c.halfSwing = 0.5f * a.angularLimits.zw;
    c.halfMinimum = -c.halfSwing; c.halfMaximum = c.halfSwing;
    c.ySin = float2(-sin(c.halfSwing.x), sin(c.halfSwing.x));
    c.zSin = float2(-sin(c.halfSwing.y), sin(c.halfSwing.y));
    c.yCos = float2(cos(c.halfSwing.x));
    c.zCos = float2(cos(c.halfSwing.y));
    const float locked = 0.008726646259971648f, free = 3.1328660073298216f;
    if (a.angularLimits.x > -locked && a.angularLimits.y < locked)
    {
        c.flags |= SwTwistLocked;
        c.twistSin = float2(0.0f); c.twistCos = float2(1.0f);
    }
    else if (a.angularLimits.x < -free && a.angularLimits.y > free)
    {
        c.flags |= SwTwistFree;
        c.twistSin = float2(-1.0f, 1.0f); c.twistCos = float2(0.0f);
    }
    if (a.angularLimits.z < locked)
    {
        c.flags |= SwYLocked;
        c.ySin = float2(0.0f); c.yCos = float2(1.0f);
    }
    else if (a.angularLimits.z > free)
    {
        c.flags |= SwYFree;
        c.ySin = float2(-1.0f, 1.0f); c.yCos = float2(0.0f);
    }
    if (a.angularLimits.w < locked)
    {
        c.flags |= SwZLocked;
        c.zSin = float2(0.0f); c.zCos = float2(1.0f);
    }
    else if (a.angularLimits.w > free)
    {
        c.flags |= SwZFree;
        c.zSin = float2(-1.0f, 1.0f); c.zCos = float2(0.0f);
    }
    return c;
}

void jph_swing_decompose(float4 q, out float4 swing, out float4 twist)
{
    float h = sqrt(q.w * q.w + q.x * q.x);
    if (h != 0.0f)
    {
        twist = float4(q.x / h, 0.0f, 0.0f, q.w / h);
        swing = float4(0.0f, (q.w * q.y - q.x * q.z) / h, (q.w * q.z + q.x * q.y) / h, h);
    }
    else
    {
        twist = float4(0.0f, 0.0f, 0.0f, 1.0f);
        swing = q;
    }
}

bool jph_swing_min_shorter(float lower, float upper)
{
    lower = abs(lower); upper = abs(upper);
    if (lower > 1.0f) lower = 2.0f - lower;
    if (upper > 1.0f) upper = 2.0f - upper;
    return lower < upper;
}

float2 jph_swing_ellipse_value(float t, float2 radius, float2 point)
{
    float2 square = radius * radius;
    float2 denominator = t + square;
    float2 value = radius * point / denominator;
    float g = dot(value, value) - 1.0f;
    float derivative = -2.0f * dot(square * point * point / (denominator * denominator * denominator), float2(1.0f));
    return float2(g, derivative);
}

bool jph_swing_ellipse(float2 radius, float2 point, out float2 closest, out float projectionT)
{
    closest = point; projectionT = 0.0f;
    float2 scaled = point / radius;
    if (dot(scaled, scaled) <= 1.0f) return true;
    float lo = 0.0f, hi = 2.0f * length(radius * point);
    if (!all(isfinite(float4(radius, point))) || any(radius <= 0.0f) ||
        !isfinite(hi) || !(hi > 0.0f) || !(jph_swing_ellipse_value(hi, radius, point).x < 0.0f))
        return false;
    float root = 0.0f;
    bool converged = false;
    for (uint iteration = 0u; iteration < 64u; ++iteration)
    {
        float mid = 0.5f * (lo + hi);
        if (mid == lo || mid == hi) break;
        float2 evaluated = jph_swing_ellipse_value(mid, radius, point);
        if (!all(isfinite(evaluated))) return false;
        if (abs(evaluated.x) < 1.0e-6f) { root = mid; converged = true; break; }
        if (evaluated.x > 0.0f) lo = mid; else hi = mid;
        float next = mid - evaluated.x / evaluated.y;
        if (isfinite(next) && next > lo && next < hi)
        {
            float value = jph_swing_ellipse_value(next, radius, point).x;
            if (!isfinite(value)) return false;
            if (abs(value) < 1.0e-6f) { root = next; converged = true; break; }
            if (value > 0.0f) lo = next; else hi = next;
        }
    }
    if (!converged && abs(jph_swing_ellipse_value(lo, radius, point).x) < 1.0e-6f)
    { root = lo; converged = true; }
    if (!converged && abs(jph_swing_ellipse_value(hi, radius, point).x) < 1.0e-6f)
    { root = hi; converged = true; }
    if (!converged) return false;
    float2 square = radius * radius;
    projectionT = root;
    closest = square * point / (root + square);
    return all(isfinite(closest));
}

// Complete only the undefined zero polar coordinate: choose angle zero when
// both component and w are exactly zero. No epsilon and no replacement axis.
float2 jph_swing_half_angles(float4 swing)
{
    float2 denominator = float2(swing.w);
    if (swing.w == 0.0f) {
        if (swing.y == 0.0f) denominator.x = 1.0f;
        if (swing.z == 0.0f) denominator.y = 1.0f;
    }
    return atan2(swing.yz, denominator);
}

// Shared Pyramid velocity normal. Cone and motor laws are unchanged.
// Real-equivalent to desired X cross current X for unit rotations, with
// the small common clamp displacement divided out before normalization.
bool jph_swing_pyramid_axis(JphSwingLimits limits, float4 world1, float4 swing, out float3 axis)
{
    axis = float3(0.0f);
    if (swing.w < 0.0f) swing = -swing;
    float2 halfAngle = jph_swing_half_angles(swing);
    float2 limited = clamp(halfAngle, limits.halfMinimum, limits.halfMaximum);
    bool2 changed = limited != halfAngle;
    if (!any(changed)) return true;

    // Preserve each unchanged polar coordinate directly. At (component,w)=0,
    // the explicit zero-coordinate completion above is sin=0, cos=1.
    // Range scaling is ordinary hypot evaluation, not an activation threshold.
    float2 range = max(abs(swing.yz), float2(abs(swing.w)));
    float2 safeRange = float2(range.x == 0.0f ? 1.0f : range.x,
                             range.y == 0.0f ? 1.0f : range.y);
    float2 scaledComponent = swing.yz / safeRange;
    float2 scaledW = float2(swing.w) / safeRange;
    float2 magnitude = range * sqrt(scaledComponent * scaledComponent + scaledW * scaledW);
    float2 safeMagnitude = float2(magnitude.x == 0.0f ? 1.0f : magnitude.x,
                                 magnitude.y == 0.0f ? 1.0f : magnitude.y);
    float2 sn = swing.yz / safeMagnitude;
    float2 cs = float2(swing.w) / safeMagnitude;
    if (magnitude.x == 0.0f) { sn.x = 0.0f; cs.x = 1.0f; }
    if (magnitude.y == 0.0f) { sn.y = 0.0f; cs.y = 1.0f; }
    if (changed.x) {
        bool lower = halfAngle.x < limits.halfMinimum.x;
        sn.x = lower ? limits.ySin.x : limits.ySin.y;
        cs.x = lower ? limits.yCos.x : limits.yCos.y;
    }
    if (changed.y) {
        bool lower = halfAngle.y < limits.halfMinimum.y;
        sn.y = lower ? limits.zSin.x : limits.zSin.y;
        cs.y = lower ? limits.zCos.x : limits.zCos.y;
    }

    float2 residual = magnitude * sin(halfAngle - limited);
    float t = max(abs(residual.x), abs(residual.y));
    if (t == 0.0f) return true;
    float2 r = residual / t;
    float3 h = float3(sn.y * r.x - sn.x * r.y, cs.y * r.x, cs.x * r.y);
    float3 g = float3(sn.x * cs.y, cs.x * sn.y, cs.x * cs.y);
    float v = dot(g, swing.yzw);
    float3 local = float3(0.0f, v * h.y - t * h.x * h.z, v * h.z + t * h.x * h.y);
    float scale = max(abs(local.x), max(abs(local.y), abs(local.z)));
    // Exact equal/antipodal X directions have no cross normal. Returning zero
    // retains that geometric degeneracy; no epsilon, prior axis or substitute.
    if (scale == 0.0f) return true;
    if (!isfinite(scale) || !all(isfinite(local)) || !all(isfinite(g))) return false;
    local = normalize(local / scale);
    float gs = max(abs(g.x), max(abs(g.y), abs(g.z)));
    if (!(gs > 0.0f)) return false;
    float4 projected = float4(0.0f, normalize(g / gs));
    float3 world = jph_rotate(jph_quat_mul(world1, projected), local);
    float lengthWorld = length(world);
    if (!(lengthWorld > 0.0f) || !isfinite(lengthWorld) || !all(isfinite(world))) return false;
    axis = world / lengthWorld;
    return true;
}

bool jph_swing_clamp(uint swingType, JphSwingLimits c,
    inout float4 swing, inout float4 twist, out uint clamped, out float coneProjectionT)
{
    clamped = 0u; coneProjectionT = 0.0f;
    bool negateSwing = swing.w < 0.0f, negateTwist = twist.w < 0.0f;
    if (negateSwing) swing = -swing;
    if (negateTwist) twist = -twist;
    if ((c.flags & SwTwistLocked) != 0u)
    {
        if (twist.x != 0.0f) clamped |= SwClampTwistMin | SwClampTwistMax;
        twist = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    else if ((c.flags & SwTwistFree) == 0u)
    {
        float lower = c.twistSin.x - twist.x, upper = twist.x - c.twistSin.y;
        if (lower > 0.0f || upper > 0.0f)
        {
            bool useLower = jph_swing_min_shorter(lower, upper);
            twist = float4(useLower ? c.twistSin.x : c.twistSin.y, 0.0f, 0.0f,
                           useLower ? c.twistCos.x : c.twistCos.y);
            clamped |= useLower ? SwClampTwistMin : SwClampTwistMax;
        }
    }
    if ((c.flags & SwYLocked) != 0u)
    {
        if ((c.flags & SwZLocked) != 0u)
        {
            if (swing.y != 0.0f) clamped |= SwClampYMin | SwClampYMax;
            if (swing.z != 0.0f) clamped |= SwClampZMin | SwClampZMax;
            swing = float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        else
        {
            if (swing.y != 0.0f) clamped |= SwClampYMin | SwClampYMax;
            float lower = c.zSin.x - swing.z, upper = swing.z - c.zSin.y;
            if (lower > 0.0f || upper > 0.0f)
            {
                bool useLower = jph_swing_min_shorter(lower, upper);
                swing = float4(0.0f, 0.0f, useLower ? c.zSin.x : c.zSin.y,
                               useLower ? c.zCos.x : c.zCos.y);
                clamped |= useLower ? SwClampZMin : SwClampZMax;
            }
            else if ((clamped & SwClampYMin) != 0u)
                swing = float4(0.0f, 0.0f, swing.z, sqrt(1.0f - swing.z * swing.z));
        }
    }
    else if ((c.flags & SwZLocked) != 0u)
    {
        if (swing.z != 0.0f) clamped |= SwClampZMin | SwClampZMax;
        float lower = c.ySin.x - swing.y, upper = swing.y - c.ySin.y;
        if (lower > 0.0f || upper > 0.0f)
        {
            bool useLower = jph_swing_min_shorter(lower, upper);
            swing = float4(0.0f, useLower ? c.ySin.x : c.ySin.y, 0.0f,
                           useLower ? c.yCos.x : c.yCos.y);
            clamped |= useLower ? SwClampYMin : SwClampYMax;
        }
        else if ((clamped & SwClampZMin) != 0u)
            swing = float4(0.0f, swing.y, 0.0f, sqrt(1.0f - swing.y * swing.y));
    }
    else if (swingType == 0u)
    {
        float2 point = swing.yz, closest;
        if (!jph_swing_ellipse(float2(c.ySin.y, c.zSin.y), point, closest, coneProjectionT)) return false;
        float2 normalizedPoint = point / float2(c.ySin.y, c.zSin.y);
        if (dot(normalizedPoint, normalizedPoint) > 1.0f)
        {
            swing = float4(0.0f, closest, sqrt(max(0.0f, 1.0f - dot(closest, closest))));
            clamped |= SwClampYMin | SwClampYMax | SwClampZMin | SwClampZMax;
        }
    }
    else
    {
        float2 halfAngle = jph_swing_half_angles(swing);
        float2 limited = clamp(halfAngle, c.halfMinimum, c.halfMaximum);
        if (any(limited != halfAngle))
        {
            float2 sn = sin(limited), cs = cos(limited);
            swing = normalize(float4(0.0f, sn.x * cs.y, cs.x * sn.y, cs.x * cs.y));
            clamped |= SwClampYMin | SwClampYMax | SwClampZMin | SwClampZMax;
        }
    }
    if (negateSwing) swing = -swing;
    if (negateTwist) twist = -twist;
    return all(isfinite(swing)) && all(isfinite(twist));
}

bool jph_swing_clamp(JointSwingAuthor a, JphSwingLimits c,
    inout float4 swing, inout float4 twist, out uint clamped, out float coneProjectionT)
{
    return jph_swing_clamp(a.angularModes.x,c,swing,twist,clamped,coneProjectionT);
}

// Exact desiredAxis x currentAxis direction with the positive projection t
// divided out algebraically. Retains the t->0 derivative without subtracting
// two almost equal world directions or disabling a genuine active cone row.
bool jph_swing_cone_axis(float4 world1, float4 swing, float4 clamped,
    float2 radius, float projectionT, out float3 axis)
{
    axis = float3(0.0f);
    // Clamp restores the input hemisphere; both quaternion signs rotate alike.
    if (swing.w < 0.0f) { swing = -swing; clamped = -clamped; }
    // At the exact PI endpoint both swings rotate X to -X. Their geometric
    // cross is exactly zero, including when float ellipse tolerance leaves
    // clamped.w at zero. Match the ordinary inactive-row outcome, not failure.
    if (swing.w == 0.0f && clamped.w == 0.0f) return true;
    float denominator = swing.w + clamped.w;
    if (!(denominator > 0.0f) || !isfinite(denominator)) return false;
    float2 deltaOverT = swing.yz / (projectionT + radius * radius);
    // Multiply the direction by positive (w+cw) before evaluation; no tiny
    // hemisphere denominator or large intermediate quaternion derivative.
    float dwNumerator = -dot(deltaOverT, swing.yz + clamped.yz);
    float ux = clamped.z * deltaOverT.x - clamped.y * deltaOverT.y;
    float2 scaledU = denominator * clamped.w * deltaOverT - clamped.yz * dwNumerator;
    float v = dot(clamped, swing);
    float3 local = float3(0.0f, v * scaledU.x - projectionT * ux * scaledU.y,
        v * scaledU.y + projectionT * ux * scaledU.x);
    float scale = max(abs(local.x), max(abs(local.y), abs(local.z)));
    if (!(scale > 0.0f) || !all(isfinite(local))) return false;
    local = normalize(local / scale);
    float3 world = jph_rotate(jph_quat_mul(world1, clamped), local);
    float size = length(world);
    if (!(size > 0.0f) || !isfinite(size) || !all(isfinite(world))) return false;
    axis = world / size;
    return true;
}

float4 jph_swing_orientation(BodyState s1, BodyState s2, JointSwingState state)
{
    return jph_quat_mul(jph_joint_conjugate(jph_quat_mul(s1.rotation, state.frame1)),
                        jph_quat_mul(s2.rotation, state.frame2));
}

// Both scalar K and apply paths use the shared projected principal inertia.
float jph_swing_inverse_mass(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, float3 axis,
    inout JointSwingState state)
{
    float k = dot(axis, jph_joint_inertia_sum(m1, s1, m2, s2, axis));
    if (k < 0.0f || !isfinite(k)) state.flags.y = JointStatusNumericalFailure;
    return k;
}

float4 jph_swing_limit_setup(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
    float3 axis, bool active, bool bilateral, inout float lambda, inout JointSwingState state)
{
    if (!active || jph_swing_inverse_mass(m1, s1, m2, s2, axis, state) == 0.0f)
    { lambda = 0.0f; return float4(0.0f); }
    return float4(axis, bilateral ? 2.0f : 1.0f);
}

float4 jph_swing_motor_setup(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
    float3 axis, uint mode, float stiffness, float damping, float cap, float friction,
    float targetVelocity, float error, float dt, inout float lambda, inout JointSwingState state)
{
    if ((mode == 0u && friction == 0.0f) || (mode == 2u && stiffness == 0.0f) ||
        (mode == 3u && stiffness == 0.0f && damping == 0.0f))
    { lambda = 0.0f; return float4(0.0f); }
    float k = jph_swing_inverse_mass(m1, s1, m2, s2, axis, state);
    if (k == 0.0f) { lambda = 0.0f; return float4(0.0f); }
    float gamma = 0.0f, bias = (mode == 1u || mode == 3u) ? -targetVelocity : 0.0f;
    if (mode == 2u || mode == 3u)
    {
        gamma = 1.0f / (dt * (damping + dt * stiffness));
        bias += dt * stiffness * gamma * error;
    }
    float4 setup = float4(1.0f / (k + gamma), gamma, bias, dt * (mode == 0u ? friction : cap));
    if (!all(isfinite(setup))) state.flags.y = JointStatusNumericalFailure;
    return setup;
}

void jph_swing_constraint_axes(uint swingType, JphSwingLimits c, float4 world1, float4 q,
    out float4 limitX, out float4 limitY, out float4 limitZ, inout uint status)
{
    limitX=limitY=limitZ=float4(0.0f);
    float4 swing, twist;
    jph_swing_decompose(q, swing, twist);
    float4 limitedSwing = swing, limitedTwist = twist;
    uint clamped; float coneProjectionT;
    if (!jph_swing_clamp(swingType, c, limitedSwing, limitedTwist, clamped, coneProjectionT))
    { status = JointStatusNumericalFailure; return; }
    float4 swung = jph_quat_mul(world1, swing);
    float3 x = jph_rotate(swung, float3(1,0,0));
    float3 y = jph_rotate(swung, float3(0,1,0));
    float3 z = jph_rotate(swung, float3(0,0,1));
    bool yActive = false, zActive = false;
    if ((c.flags & SwYLocked) != 0u)
    {
        yActive = true;
        zActive = (c.flags & SwZLocked) != 0u || (clamped & (SwClampZMin | SwClampZMax)) != 0u;
        if ((c.flags & SwZLocked) == 0u && (clamped & SwClampZMin) != 0u) z = -z;
    }
    else if ((c.flags & SwZLocked) != 0u)
    {
        zActive = true;
        yActive = (clamped & (SwClampYMin | SwClampYMax)) != 0u;
        if ((clamped & SwClampYMin) != 0u) y = -y;
    }
    else if ((c.flags & (SwYFree | SwZFree)) != (SwYFree | SwZFree) &&
             (clamped & (SwClampYMin | SwClampYMax | SwClampZMin | SwClampZMax)) != 0u)
    {
        if (swingType == 0u)
        {
            if (!jph_swing_cone_axis(world1, swing, limitedSwing,
                float2(c.ySin.y, c.zSin.y), coneProjectionT, y))
            { status = JointStatusNumericalFailure; return; }
            yActive = any(y != float3(0.0f));
        }
        else
        {
            if (!jph_swing_pyramid_axis(c, world1, swing, y))
            { status = JointStatusNumericalFailure; return; }
            yActive = any(y != float3(0.0f));
        }
    }
    bool xActive = (c.flags & SwTwistLocked) != 0u ||
        ((c.flags & SwTwistFree) == 0u && (clamped & (SwClampTwistMin | SwClampTwistMax)) != 0u);
    if ((c.flags & SwTwistLocked) == 0u && (clamped & SwClampTwistMin) != 0u) x = -x;
    limitX=float4(x,xActive?(c.twistSin.x==c.twistSin.y?2.0f:1.0f):0.0f);
    limitY=float4(y,yActive?(c.ySin.x==c.ySin.y?2.0f:1.0f):0.0f);
    limitZ=float4(z,zActive?(c.zSin.x==c.zSin.y?2.0f:1.0f):0.0f);
}

void jph_swing_setup(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, float dt)
{
    if ((state.flags.w & JointPoseDriveActive) != 0u)
        a.angularTarget = normalize(jph_quat_mul(jph_quat_mul(jph_joint_conjugate(state.frame1), state.poseDriveRelative), state.frame2));

    JphSwingLimits c = jph_swing_limits(a);
    float4 world1 = jph_quat_mul(s1.rotation, state.frame1);
    float4 world2 = jph_quat_mul(s2.rotation, state.frame2);
    float4 q = jph_quat_mul(jph_joint_conjugate(world1), world2);
    float4 axisX,axisY,axisZ;
    jph_swing_constraint_axes(a.angularModes.x,c,world1,q,axisX,axisY,axisZ,state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    float lx = state.angularLimitImpulse.x, ly = state.angularLimitImpulse.y, lz = state.angularLimitImpulse.z;
    state.angularLimitX = jph_swing_limit_setup(m1,s1,m2,s2,axisX.xyz,axisX.w!=0.0f,axisX.w==2.0f,lx,state);
    state.angularLimitY = jph_swing_limit_setup(m1,s1,m2,s2,axisY.xyz,axisY.w!=0.0f,axisY.w==2.0f,ly,state);
    state.angularLimitZ = jph_swing_limit_setup(m1,s1,m2,s2,axisZ.xyz,axisZ.w!=0.0f,axisZ.w==2.0f,lz,state);
    state.angularLimitImpulse = float4(lx,ly,lz,0.0f);

    float3 error = float3(0.0f);
    if (a.angularModes.y >= 2u || a.angularModes.z >= 2u)
    {
        float4 targetSwing, targetTwist;
        jph_swing_decompose(a.angularTarget, targetSwing, targetTwist);
        uint targetClamped; float targetConeProjectionT;
        if (!jph_swing_clamp(a,c,targetSwing,targetTwist,targetClamped,targetConeProjectionT))
        { state.flags.y = JointStatusNumericalFailure; return; }
        float4 target = targetClamped != 0u ? jph_quat_mul(targetSwing,targetTwist) : a.angularTarget;
        if (dot(q,target) <= 0.0f) target = -target;
        error = -2.0f * jph_quat_mul(jph_joint_conjugate(q),target).xyz;
    }
    // Exact fork behavior: Off Swing with no friction deactivates all motor
    // parts, then the twist branch sets X up again with a cleared warm lambda.
    if (a.angularModes.y == 0u && a.angularTorque.z == 0.0f)
        state.angularMotorImpulse = float4(0.0f);
    float mx=state.angularMotorImpulse.x, my=state.angularMotorImpulse.y, mz=state.angularMotorImpulse.z;
    state.angularMotorSetupX = jph_swing_motor_setup(m1,s1,m2,s2,jph_rotate(world2,float3(1,0,0)),
        a.angularModes.z,a.angularSpring.z,a.angularSpring.w,a.angularTorque.y,a.angularTorque.z,
        a.angularVelocity.x,error.x,dt,mx,state);
    state.angularMotorSetupY = jph_swing_motor_setup(m1,s1,m2,s2,jph_rotate(world2,float3(0,1,0)),
        a.angularModes.y,a.angularSpring.x,a.angularSpring.y,a.angularTorque.x,a.angularTorque.z,
        a.angularVelocity.y,error.y,dt,my,state);
    state.angularMotorSetupZ = jph_swing_motor_setup(m1,s1,m2,s2,jph_rotate(world2,float3(0,0,1)),
        a.angularModes.y,a.angularSpring.x,a.angularSpring.y,a.angularTorque.x,a.angularTorque.z,
        a.angularVelocity.z,error.z,dt,mz,state);
    state.angularMotorImpulse = float4(mx,my,mz,0.0f);
}

void jph_swing_warm(JointSwingAuthor a, JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    float4 world2 = jph_quat_mul(s2.rotation,state.frame2);
    jph_joint_apply_angular(m1,s1,m2,s2,jph_rotate(world2,float3(1,0,0))*state.angularMotorImpulse.x,false);
    jph_joint_apply_angular(m1,s1,m2,s2,jph_rotate(world2,float3(0,1,0))*state.angularMotorImpulse.y,false);
    jph_joint_apply_angular(m1,s1,m2,s2,jph_rotate(world2,float3(0,0,1))*state.angularMotorImpulse.z,false);
    jph_joint_apply_angular(m1,s1,m2,s2,state.angularLimitY.xyz*state.angularLimitImpulse.y,false);
    jph_joint_apply_angular(m1,s1,m2,s2,state.angularLimitZ.xyz*state.angularLimitImpulse.z,false);
    jph_joint_apply_angular(m1,s1,m2,s2,state.angularLimitX.xyz*state.angularLimitImpulse.x,false);
    float3 r1,r2;
    jph_joint_arms(a,state,s1,s2,r1,r2);
    jph_joint_apply_point(m1,s1,r1,m2,s2,r2,state.pointImpulse.xyz,false);
}

void jph_swing_motor_velocity(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
    float3 axis, float4 setup, inout float total)
{
    if (setup.x == 0.0f) return;
    float jv = dot(axis,jph_joint_angular_velocity(m1,s1)-jph_joint_angular_velocity(m2,s2));
    float next = clamp(total + setup.x*(jv-setup.y*total-setup.z),-setup.w,setup.w);
    jph_joint_apply_angular(m1,s1,m2,s2,axis*(next-total),false);
    total = next;
}

void jph_swing_limit_velocity(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
    float4 axis, inout float total, inout JointSwingState state)
{
    if (axis.w == 0.0f) return;
    float k = jph_swing_inverse_mass(m1,s1,m2,s2,axis.xyz,state);
    if (!(k > 0.0f) || state.flags.y != JointStatusSuccess) return;
    float next = total + dot(axis.xyz,jph_joint_angular_velocity(m1,s1)-jph_joint_angular_velocity(m2,s2))/k;
    if (axis.w == 1.0f) next = min(0.0f,next);
    jph_joint_apply_angular(m1,s1,m2,s2,axis.xyz*(next-total),false);
    total = next;
}

void jph_swing_velocity(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    float4 world2 = jph_quat_mul(s2.rotation,state.frame2);
    float mx=state.angularMotorImpulse.x, my=state.angularMotorImpulse.y, mz=state.angularMotorImpulse.z;
    jph_swing_motor_velocity(m1,s1,m2,s2,jph_rotate(world2,float3(1,0,0)),state.angularMotorSetupX,mx);
    jph_swing_motor_velocity(m1,s1,m2,s2,jph_rotate(world2,float3(0,1,0)),state.angularMotorSetupY,my);
    jph_swing_motor_velocity(m1,s1,m2,s2,jph_rotate(world2,float3(0,0,1)),state.angularMotorSetupZ,mz);
    state.angularMotorImpulse = float4(mx,my,mz,0.0f);
    float lx=state.angularLimitImpulse.x, ly=state.angularLimitImpulse.y, lz=state.angularLimitImpulse.z;
    jph_swing_limit_velocity(m1,s1,m2,s2,state.angularLimitY,ly,state);
    jph_swing_limit_velocity(m1,s1,m2,s2,state.angularLimitZ,lz,state);
    jph_swing_limit_velocity(m1,s1,m2,s2,state.angularLimitX,lx,state);
    state.angularLimitImpulse = float4(lx,ly,lz,0.0f);
    float3 r1,r2;
    jph_joint_arms(a,state,s1,s2,r1,r2);
    float3 point = jph_joint_solve_point(m1,s1,r1,m2,s2,r2,
        jph_point_velocity(m1,s1,r1)-jph_point_velocity(m2,s2,r2), state.flags.y);
    state.pointImpulse.xyz += point;
    jph_joint_apply_point(m1,s1,r1,m2,s2,r2,point,false);
}

void jph_swing_constraint_position(uint swingType, JphSwingLimits c, float4 q, float4 frame1, float4 frame2,
    bool sixCut,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte, inout uint status)
{
    float4 swing,twist;
    jph_swing_decompose(q,swing,twist);
    uint clamped; float coneProjectionT;
    if (!jph_swing_clamp(swingType,c,swing,twist,clamped,coneProjectionT))
    { status = JointStatusNumericalFailure; return; }
    if (clamped != 0u)
    {
        float4 inverseInitial = jph_quat_mul(frame2,
            jph_joint_conjugate(jph_quat_mul(frame1,jph_quat_mul(swing,twist))));
        float4 diff = jph_quat_mul(jph_quat_mul(s2.rotation,inverseInitial),jph_joint_conjugate(s1.rotation));
        if (sixCut) diff = jph_six_cut(diff);
        float3 error = 2.0f*(diff.w < 0.0f ? -diff.xyz : diff.xyz);
        float3 angular = jph_joint_solve_rotation(m1,s1,m2,s2,-baumgarte*error, status);
        if (status != JointStatusSuccess) return;
        jph_joint_apply_angular(m1,s1,m2,s2,angular,true);
    }
}

void jph_swing_position(JointSwingAuthor a, inout JointSwingState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    jph_swing_constraint_position(a.angularModes.x,jph_swing_limits(a),jph_swing_orientation(s1,s2,state),state.frame1,state.frame2,false,
        m1,s1,m2,s2,baumgarte,state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    float3 r1,r2;
    jph_joint_arms(a,state,s1,s2,r1,r2);
    float3 u = s2.positionCOM.xyz-s1.positionCOM.xyz+r2-r1;
    float3 point = jph_joint_solve_point(m1,s1,r1,m2,s2,r2,-baumgarte*u, state.flags.y);
    if (state.flags.y != JointStatusSuccess) return;
    jph_joint_apply_point(m1,s1,r1,m2,s2,r2,point,true);
}

#endif // JPH_GPU_RIGID_SWING_TWIST_SLANG
