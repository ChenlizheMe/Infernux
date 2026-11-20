// Jolt SixDOFConstraint law; included after shared joint/Swing math. No resources.
#ifndef JPH_GPU_RIGID_SIX_DOF_SLANG
#define JPH_GPU_RIGID_SIX_DOF_SLANG

uint jph_six_mode(uint packed, uint axis)
{
    return (packed >> (2u * axis)) & 3u;
}
bool jph_six_point_block(JointSixAuthor a)
{
    return (a.sixModes.x & 63u) == 0u && all(a.sixLimitStiffness.xyz == float3(0.0f));
}
bool jph_six_rotation_block(JointSixAuthor a)
{
    return ((a.sixModes.x >> 6u) & 63u) == 0u;
}
float2 jph_six_bounds(uint mode, float lo, float hi, bool angular)
{
    if (mode == 0u)
        return float2(0.0f);
    if (mode == 2u)
        return angular ? float2(-3.14159265358979323846f, 3.14159265358979323846f)
                       : float2(-3.4028234663852886e38f, 3.4028234663852886e38f);
    return float2(lo, hi);
}
JphSwingLimits jph_six_angular_limits(JointSixAuthor a)
{
    float2 x = jph_six_bounds(jph_six_mode(a.sixModes.x, 3u), a.sixAngularMin.x, a.sixAngularMax.x, true);
    float2 y = jph_six_bounds(jph_six_mode(a.sixModes.x, 4u), a.sixAngularMin.y, a.sixAngularMax.y, true);
    float2 z = jph_six_bounds(jph_six_mode(a.sixModes.x, 5u), a.sixAngularMin.z, a.sixAngularMax.z, true);
    JphSwingLimits c;
    c.flags = 0u;
    c.twistSin = sin(0.5f * x);
    c.twistCos = cos(0.5f * x);
    c.ySin = sin(0.5f * y);
    c.yCos = cos(0.5f * y);
    c.zSin = sin(0.5f * z);
    c.zCos = cos(0.5f * z);
    c.halfMinimum = 0.5f * float2(y.x, z.x);
    c.halfMaximum = 0.5f * float2(y.y, z.y);
    c.halfSwing = c.halfMaximum;
    const float locked = 0.008726646259971648f, free = 3.1328660073298216f;
    if (x.x > -locked && x.y < locked) {
        c.flags |= SwTwistLocked;
        c.twistSin = float2(0);
        c.twistCos = float2(1);
    } else if (x.x < -free && x.y > free) {
        c.flags |= SwTwistFree;
        c.twistSin = float2(-1, 1);
        c.twistCos = float2(0);
    }
    if (y.x > -locked && y.y < locked) {
        c.flags |= SwYLocked;
        c.ySin = float2(0);
        c.yCos = float2(1);
    } else if (y.x < -free && y.y > free) {
        c.flags |= SwYFree;
        c.ySin = float2(-1, 1);
        c.yCos = float2(0);
    }
    if (z.x > -locked && z.y < locked) {
        c.flags |= SwZLocked;
        c.zSin = float2(0);
        c.zCos = float2(1);
    } else if (z.x < -free && z.y > free) {
        c.flags |= SwZFree;
        c.zSin = float2(-1, 1);
        c.zCos = float2(0);
    }
    return c;
}
float4 jph_six_orientation(BodyState s1, BodyState s2, JointSixState state)
{
    return jph_six_cut(jph_quat_mul(jph_joint_conjugate(jph_quat_mul(s1.rotation, state.frame1)),
                                  jph_quat_mul(s2.rotation, state.frame2)));
}
float3 jph_six_axis(float4 frame, uint i)
{
    return jph_rotate(frame, i == 0u ? float3(1, 0, 0) : i == 1u ? float3(0, 1, 0) : float3(0, 0, 1));
}
void jph_six_geometry(JointSixAuthor a, JointSixState state, BodyState s1, BodyState s2, out float3 arm1,
                      out float3 arm2, out float3 u)
{
    float3 r1;
    jph_joint_arms(a, state, s1, s2, r1, arm2);
    u = s2.positionCOM.xyz - s1.positionCOM.xyz + arm2 - r1;
    arm1 = r1 + u;
}
float jph_six_direction_support(BodyModel m, BodyState s, float3 arm, float3 axis, bool point)
{
    if (!jph_dynamic(m))
        return 0.0f;
    float rowScale = max(abs(axis.x), max(abs(axis.y), abs(axis.z)));
    if (rowScale == 0.0f)
        return 0.0f;
    axis /= rowScale;
    float armScale = max(abs(arm.x), max(abs(arm.y), abs(arm.z)));
    if (point && armScale != 0.0f)
        arm /= armScale;
    // Mobility rank follows world DOFs, not inertia eigenvectors.
    float support = 0.0f;
    for (uint i = 0u; i < 3u; ++i) {
        if (point && (uint(m.mass.y) & (1u << i)) == 0u)
            support = max(support, abs(axis[i]));
        if ((uint(m.mass.z) & (1u << i)) == 0u) {
            float3 column = float3(0.0f); column[i] = 1.0f;
            if (point)
                column = cross(column, arm);
            support = max(support, abs(dot(axis, column)));
        }
    }
    return support;
}
float jph_six_body_inverse_mass(BodyModel m, BodyState s, float3 arm, float3 axis, bool point)
{
    if (!jph_dynamic(m))
        return 0.0f;
    float3 angular = point ? cross(arm, axis) : axis;
    // Pworld acts on the Jacobian before the physical principal transform.
    // Evaluate its positive quadratic form without world-space reconstruction
    // and cancellation; retain weight-before-square for weak inertia/lever arms.
    angular = jph_inverse_rotate(jph_quat_mul(s.rotation, m.inertiaRotation),
        angular * jph_allowed_axes(m.mass.z));
    float3 weighted = angular * sqrt(m.invInertiaDiagonal.xyz);
    return (point ? m.mass.x * dot(axis * axis, jph_allowed_axes(m.mass.y)) : 0.0f) + dot(weighted, weighted);
}
float jph_six_inverse_mass(BodyModel m1, BodyState s1, float3 arm1, BodyModel m2, BodyState s2, float3 arm2,
                           float3 axis, bool point, inout uint status)
{
    if (!all(isfinite(arm1)) || !all(isfinite(arm2)) || !all(isfinite(axis))) {
        status = JointStatusNumericalFailure;
        return 0.0f;
    }
    // Same dimensionless float32 direction resolution as practicalQR. Each
    // finite body's lever is scaled independently only for this range test.
    float support =
        max(jph_six_direction_support(m1, s1, arm1, axis, point), jph_six_direction_support(m2, s2, arm2, axis, point));
    if (!isfinite(support)) {
        status = JointStatusNumericalFailure;
        return 0.0f;
    }
    if (support <= 8.0f * 1.1920928955078125e-7f)
        return 0.0f;
    float k =
        jph_six_body_inverse_mass(m1, s1, arm1, axis, point) + jph_six_body_inverse_mass(m2, s2, arm2, axis, point);
    if (!isfinite(k) || k < 0.0f)
        status = JointStatusNumericalFailure;
    return k;
}
float4 jph_six_angular_limit(float4 axis, BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, inout float lambda,
                             inout uint status)
{
    if (axis.w == 0.0f) {
        lambda = 0.0f;
        return float4(0);
    }
    float k = jph_six_inverse_mass(m1, s1, float3(0), m2, s2, float3(0), axis.xyz, false, status);
    if (k == 0.0f) {
        lambda = 0.0f;
        return float4(0);
    }
    // Six alone caches signed Meff: positive bilateral, negative unilateral.
    // The old Swing state encoding and arithmetic remain unchanged.
    return float4(axis.xyz, (axis.w == 2.0f ? 1.0f : -1.0f) / k);
}
float4 jph_six_spring_setup(float inverseMass, float stiffness, float damping, float error, float velocityBias,
                            float dt, float cap, inout uint status)
{
    float gamma = 1.0f / (dt * (damping + dt * stiffness));
    float4 setup = float4(1.0f / (inverseMass + gamma), gamma, velocityBias + dt * stiffness * gamma * error, cap);
    if (!all(isfinite(setup)))
        status = JointStatusNumericalFailure;
    return setup;
}
float jph_six_limit(uint mode, float2 bounds, float d, out float error)
{
    error = 0.0f;
    if (mode == 0u) {
        error = d;
        return 2.0f;
    }
    if (mode == 2u)
        return 0.0f;
    if (d <= bounds.x) {
        error = d - bounds.x;
        return -1.0f;
    }
    if (d >= bounds.y) {
        error = d - bounds.y;
        return 1.0f;
    }
    return 0.0f;
}
float4 jph_six_twist(float4 q, uint axis)
{
    float4 projected = float4(0, 0, 0, q.w);
    projected[axis] = q[axis];
    float lengthSq = dot(projected, projected);
    return lengthSq != 0.0f ? projected / sqrt(lengthSq) : float4(0, 0, 0, 1);
}
float3 jph_six_motor_error(JointSixAuthor a, JphSwingLimits c, float4 q, inout uint status)
{
    float4 swing, twist;
    float4 authoredTarget = jph_six_cut(a.angularTarget);
    jph_swing_decompose(authoredTarget, swing, twist);
    uint clamped;
    float unused;
    if (!jph_swing_clamp(1u, c, swing, twist, clamped, unused)) {
        status = JointStatusNumericalFailure;
        return float3(0);
    }
    float4 target = clamped != 0u ? jph_quat_mul(swing, twist) : authoredTarget;
    if (dot(q, target) <= 0.0f)
        target = -target;
    float4 diff = jph_quat_mul(jph_joint_conjugate(q), target);
    uint mask = 0u;
    for (uint i = 0u; i < 3u; ++i)
        if (jph_six_mode(a.sixModes.y, i + 3u) >= 2u)
            mask |= 1u << i;
    if (mask == 1u || mask == 2u || mask == 4u)
        diff = jph_six_twist(diff, mask == 1u ? 0u : mask == 2u ? 1u : 2u);
    else if (mask == 3u || mask == 5u || mask == 6u)
        diff = jph_quat_mul(diff, jph_joint_conjugate(jph_six_twist(diff, mask == 3u ? 2u : mask == 5u ? 1u : 0u)));
    return -2.0f * diff.xyz;
}
void jph_joint_setup(JointSixAuthor a, inout JointSixState state, BodyModel m1, inout BodyState s1, BodyModel m2,
                     inout BodyState s2, float dt, bool warm_start)
{
    if ((state.flags.w & JointPoseDriveActive) != 0u)
        a.angularTarget = normalize(jph_quat_mul(jph_quat_mul(jph_joint_conjugate(state.frame1), state.poseDriveRelative), state.frame2));

    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2))) {
        jph_joint_clear_impulses(state);
        return;
    }
    float4 world1 = jph_quat_mul(s1.rotation, state.frame1), world2 = jph_quat_mul(s2.rotation, state.frame2);
    float3 arm1, arm2, u;
    jph_six_geometry(a, state, s1, s2, arm1, arm2, u);
    float4 axes[3], limits[3], motors[3];
    for (uint i = 0u; i < 3u; ++i) {
        float3 axis = jph_six_axis(world1, i);
        float d = dot(axis, u);
        axes[i] = float4(axis, d);
        limits[i] = motors[i] = float4(0);
        if (jph_six_point_block(a)) {
            state.sixLimitImpulse[i] = state.sixMotorImpulse[i] = 0.0f;
            continue;
        }
        uint motion = jph_six_mode(a.sixModes.x, i), motor = jph_six_mode(a.sixModes.y, i);
        float error;
        float side =
            jph_six_limit(motion, jph_six_bounds(motion, a.sixLinearMin[i], a.sixLinearMax[i], false), d, error);
        bool active = motor == 0u   ? (motion != 0u && a.sixLinearFriction[i] > 0.0f)
                      : motor == 1u ? true
                      : motor == 2u ? a.sixLinearMotorK[i] > 0.0f
                                    : (a.sixLinearMotorK[i] > 0.0f || a.sixLinearMotorC[i] > 0.0f);
        if (side == 0.0f && !active) {
            state.sixLimitImpulse[i] = state.sixMotorImpulse[i] = 0.0f;
            continue;
        }
        float k = jph_six_inverse_mass(m1, s1, arm1, m2, s2, arm2, axis, true, state.flags.y);
        // Compliance is added only after a resolvable physical mobility exists,
        // matching AxisConstraintPart's deactivate-before-spring ordering.
        if (side != 0.0f && k > 0.0f) {
            if (a.sixLimitStiffness[i] > 0.0f)
                limits[i] = jph_six_spring_setup(k, a.sixLimitStiffness[i], a.sixLimitDamping[i], error, 0.0f, dt, side,
                                                 state.flags.y);
            else
                limits[i] = float4(1.0f / k, 0, 0, side);
        } else
            state.sixLimitImpulse[i] = 0.0f;
        if (active && k > 0.0f) {
            float cap = dt * (motor == 0u ? a.sixLinearFriction[i] : a.sixLinearMotorForce[i]);
            float bias = (motor == 1u || motor == 3u) ? -a.sixTargetVelocity[i] : 0.0f;
            motors[i] = motor >= 2u ? jph_six_spring_setup(k, a.sixLinearMotorK[i], a.sixLinearMotorC[i],
                                                           d - a.sixTargetPosition[i], bias, dt, cap, state.flags.y)
                                    : float4(1.0f / k, 0, bias, cap);
        } else
            state.sixMotorImpulse[i] = 0.0f;
    }
    state.sixTranslationAxisX = axes[0];
    state.sixTranslationAxisY = axes[1];
    state.sixTranslationAxisZ = axes[2];
    state.sixLimitSetupX = limits[0];
    state.sixLimitSetupY = limits[1];
    state.sixLimitSetupZ = limits[2];
    state.sixMotorSetupX = motors[0];
    state.sixMotorSetupY = motors[1];
    state.sixMotorSetupZ = motors[2];
    if (jph_six_rotation_block(a)) {
        state.angularLimitImpulse = state.angularMotorImpulse = float4(0);
        state.angularLimitX = state.angularLimitY = state.angularLimitZ = float4(0);
        state.angularMotorSetupX = state.angularMotorSetupY = state.angularMotorSetupZ = float4(0);
    } else {
        JphSwingLimits c = jph_six_angular_limits(a);
        float4 q = jph_six_cut(jph_quat_mul(jph_joint_conjugate(world1), world2));
        float4 axisX, axisY, axisZ;
        jph_swing_constraint_axes(1u, c, world1, q, axisX, axisY, axisZ, state.flags.y);
        float lx = state.angularLimitImpulse.x, ly = state.angularLimitImpulse.y, lz = state.angularLimitImpulse.z;
        state.angularLimitX = jph_six_angular_limit(axisX, m1, s1, m2, s2, lx, state.flags.y);
        state.angularLimitY = jph_six_angular_limit(axisY, m1, s1, m2, s2, ly, state.flags.y);
        state.angularLimitZ = jph_six_angular_limit(axisZ, m1, s1, m2, s2, lz, state.flags.y);
        state.angularLimitImpulse = float4(lx, ly, lz, 0);
        float3 error = jph_six_motor_error(a, c, q, state.flags.y);
        float4 setups[3];
        for (uint i = 0u; i < 3u; ++i) {
            uint motor = jph_six_mode(a.sixModes.y, i + 3u);
            float friction = jph_six_mode(a.sixModes.x, i + 3u) != 0u ? a.sixAngularFriction[i] : 0.0f;
            float lambda = state.angularMotorImpulse[i];
            bool active = motor == 0u   ? friction > 0.0f
                          : motor == 1u ? true
                          : motor == 2u ? a.sixAngularMotorK[i] > 0.0f
                                        : (a.sixAngularMotorK[i] > 0.0f || a.sixAngularMotorC[i] > 0.0f);
            float k = 0.0f;
            if (active)
                k = jph_six_inverse_mass(m1, s1, float3(0), m2, s2, float3(0), jph_six_axis(world2, i), false,
                                         state.flags.y);
            setups[i] = float4(0);
            if (active && k > 0.0f) {
                float bias = (motor == 1u || motor == 3u) ? -a.angularVelocity[i] : 0.0f;
                float cap = dt * (motor == 0u ? friction : a.sixAngularMotorTorque[i]);
                setups[i] = motor >= 2u ? jph_six_spring_setup(k, a.sixAngularMotorK[i], a.sixAngularMotorC[i],
                                                               error[i], bias, dt, cap, state.flags.y)
                                        : float4(1.0f / k, 0, bias, cap);
            } else
                lambda = 0.0f;
            state.angularMotorImpulse[i] = lambda;
        }
        state.angularMotorSetupX = setups[0];
        state.angularMotorSetupY = setups[1];
        state.angularMotorSetupZ = setups[2];
    }
    if (!warm_start) {
        jph_joint_clear_impulses(state);
        return;
    }
    float previous = asfloat(state.flags.z);
    float ratio = previous > 0.0f ? dt / previous : 0.0f;
    state.pointImpulse *= ratio;
    state.angularImpulse *= ratio;
    state.angularLimitImpulse *= ratio;
    state.angularMotorImpulse *= ratio;
    state.sixLimitImpulse *= ratio;
    state.sixMotorImpulse *= ratio;
    if (jph_six_point_block(a)) {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        state.pointImpulse.xyz =
            jph_joint_project_range(jph_joint_reachable(m1, s1, r1, m2, s2, r2, true, 3u, float3(0), float3(0)),
                                    state.pointImpulse.xyz, 3u, state.flags.y);
    } else
        state.pointImpulse = float4(0);
    if (jph_six_rotation_block(a))
        state.angularImpulse.xyz = jph_joint_project_range(
            jph_joint_reachable(m1, s1, float3(0), m2, s2, float3(0), false, 3u, float3(0), float3(0)),
            state.angularImpulse.xyz, 3u, state.flags.y);
    else
        state.angularImpulse = float4(0);
}
void jph_six_translation_velocity(BodyModel m1, inout BodyState s1, float3 arm1, BodyModel m2, inout BodyState s2,
                                  float3 arm2, float3 axis, float4 setup, bool limit, inout float lambda)
{
    if (setup.x == 0.0f)
        return;
    float jv = dot(axis, jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, arm2));
    float next = lambda + setup.x * (jv - setup.z - setup.y * lambda);
    if (limit) {
        if (setup.w == -1.0f)
            next = max(0.0f, next);
        else if (setup.w == 1.0f)
            next = min(0.0f, next);
    } else
        next = clamp(next, -setup.w, setup.w);
    jph_joint_apply_point(m1, s1, arm1, m2, s2, arm2, axis * (next - lambda), false);
    lambda = next;
}
void jph_six_angular_velocity(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float4 row,
                              inout float lambda)
{
    if (row.w == 0.0f)
        return;
    float next =
        lambda + abs(row.w) * dot(row.xyz, jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2));
    if (row.w < 0.0f)
        next = min(0.0f, next);
    jph_joint_apply_angular(m1, s1, m2, s2, row.xyz * (next - lambda), false);
    lambda = next;
}
void jph_joint_warm_start(JointSixAuthor a, JointSixState state, BodyModel m1, inout BodyState s1, BodyModel m2,
                          inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess || (a.bodies.w & JointEnabled) == 0u)
        return;
    float3 arm1, arm2, u;
    jph_six_geometry(a, state, s1, s2, arm1, arm2, u);
    float4 axes[3] = {state.sixTranslationAxisX, state.sixTranslationAxisY, state.sixTranslationAxisZ};
    float4 world2 = jph_quat_mul(s2.rotation, state.frame2);
    for (uint i = 0u; i < 3u; ++i)
        jph_joint_apply_point(m1, s1, arm1, m2, s2, arm2, axes[i].xyz * state.sixMotorImpulse[i], false);
    for (uint i = 0u; i < 3u; ++i)
        jph_joint_apply_angular(m1, s1, m2, s2, jph_six_axis(world2, i) * state.angularMotorImpulse[i], false);
    if (jph_six_rotation_block(a))
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularImpulse.xyz, false);
    else {
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularLimitY.xyz * state.angularLimitImpulse.y, false);
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularLimitZ.xyz * state.angularLimitImpulse.z, false);
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularLimitX.xyz * state.angularLimitImpulse.x, false);
    }
    if (jph_six_point_block(a)) {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, state.pointImpulse.xyz, false);
    } else
        for (uint i = 0u; i < 3u; ++i)
            jph_joint_apply_point(m1, s1, arm1, m2, s2, arm2, axes[i].xyz * state.sixLimitImpulse[i], false);
}
void jph_joint_solve_velocity(JointSixAuthor a, inout JointSixState state, BodyModel m1, inout BodyState s1,
                              BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess || (a.bodies.w & JointEnabled) == 0u)
        return;
    float3 arm1, arm2, u;
    jph_six_geometry(a, state, s1, s2, arm1, arm2, u);
    float4 axes[3] = {state.sixTranslationAxisX, state.sixTranslationAxisY, state.sixTranslationAxisZ};
    float4 motors[3] = {state.sixMotorSetupX, state.sixMotorSetupY, state.sixMotorSetupZ};
    float4 limits[3] = {state.sixLimitSetupX, state.sixLimitSetupY, state.sixLimitSetupZ};
    for (uint i = 0u; i < 3u; ++i) {
        float lambda = state.sixMotorImpulse[i];
        jph_six_translation_velocity(m1, s1, arm1, m2, s2, arm2, axes[i].xyz, motors[i], false, lambda);
        state.sixMotorImpulse[i] = lambda;
    }
    float4 world2 = jph_quat_mul(s2.rotation, state.frame2);
    float4 angularMotors[3] = {state.angularMotorSetupX, state.angularMotorSetupY, state.angularMotorSetupZ};
    for (uint i = 0u; i < 3u; ++i) {
        float lambda = state.angularMotorImpulse[i];
        jph_swing_motor_velocity(m1, s1, m2, s2, jph_six_axis(world2, i), angularMotors[i], lambda);
        state.angularMotorImpulse[i] = lambda;
    }
    if (jph_six_rotation_block(a)) {
        float3 lambda = jph_joint_solve_rotation(
            m1, s1, m2, s2, jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2), state.flags.y);
        state.angularImpulse.xyz += lambda;
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, false);
    } else {
        float x = state.angularLimitImpulse.x, y = state.angularLimitImpulse.y, z = state.angularLimitImpulse.z;
        jph_six_angular_velocity(m1, s1, m2, s2, state.angularLimitY, y);
        jph_six_angular_velocity(m1, s1, m2, s2, state.angularLimitZ, z);
        jph_six_angular_velocity(m1, s1, m2, s2, state.angularLimitX, x);
        state.angularLimitImpulse = float4(x, y, z, 0);
    }
    if (jph_six_point_block(a)) {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        float3 lambda = jph_joint_solve_point(
            m1, s1, r1, m2, s2, r2, jph_point_velocity(m1, s1, r1) - jph_point_velocity(m2, s2, r2), state.flags.y);
        state.pointImpulse.xyz += lambda;
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, lambda, false);
    } else
        for (uint i = 0u; i < 3u; ++i) {
            float lambda = state.sixLimitImpulse[i];
            jph_six_translation_velocity(m1, s1, arm1, m2, s2, arm2, axes[i].xyz, limits[i], true, lambda);
            state.sixLimitImpulse[i] = lambda;
        }
}
void jph_joint_solve_position(JointSixAuthor a, inout JointSixState state, BodyModel m1, inout BodyState s1,
                              BodyModel m2, inout BodyState s2, float beta)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess || (a.bodies.w & JointEnabled) == 0u)
        return;
    if (jph_six_rotation_block(a)) {
        float4 inverseInitial = jph_quat_mul(state.frame2, jph_joint_conjugate(state.frame1));
        float4 diff = jph_six_cut(jph_quat_mul(jph_quat_mul(s2.rotation, inverseInitial), jph_joint_conjugate(s1.rotation)));
        float3 error = 2.0f * (diff.w < 0.0f ? -diff.xyz : diff.xyz);
        float3 lambda = jph_joint_solve_rotation(m1, s1, m2, s2, -beta * error, state.flags.y);
        if (state.flags.y != JointStatusSuccess)
            return;
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, true);
    } else
        jph_swing_constraint_position(1u, jph_six_angular_limits(a), jph_six_orientation(s1, s2, state), state.frame1,
                                      state.frame2, true, m1, s1, m2, s2, beta, state.flags.y);
    if (state.flags.y != JointStatusSuccess)
        return;
    if (jph_six_point_block(a)) {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        float3 lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2, -beta * u, state.flags.y);
        if (state.flags.y != JointStatusSuccess)
            return;
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, lambda, true);
    } else
        for (uint i = 0u; i < 3u; ++i) {
            if (a.sixLimitStiffness[i] > 0.0f)
                continue;
            uint motion = jph_six_mode(a.sixModes.x, i);
            if (motion == 2u)
                continue;
            float3 arm1, arm2, u;
            jph_six_geometry(a, state, s1, s2, arm1, arm2, u);
            float3 axis = jph_six_axis(jph_quat_mul(s1.rotation, state.frame1), i);
            float error;
            float side = jph_six_limit(motion, jph_six_bounds(motion, a.sixLinearMin[i], a.sixLinearMax[i], false),
                                       dot(axis, u), error);
            if (side == 0.0f || error == 0.0f)
                continue;
            float k = jph_six_inverse_mass(m1, s1, arm1, m2, s2, arm2, axis, true, state.flags.y);
            if (state.flags.y != JointStatusSuccess)
                return;
            if (k > 0.0f)
                jph_joint_apply_point(m1, s1, arm1, m2, s2, arm2, axis * (-beta * error / k), true);
        }
}
#endif
