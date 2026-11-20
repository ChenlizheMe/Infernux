// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_RIGID_MATH_SLANG
#define JPH_GPU_RIGID_MATH_SLANG

#include "../PhysicsTypes.h"

// Pure math, with the same xyzw / world angular velocity conventions as
// Jolt/Shaders/ShaderQuat.h and Physics/Body/Body.inl.
float4 jph_quat_mul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                  a.w * b.w - dot(a.xyz, b.xyz));
}

float3 jph_rotate(float4 q, float3 v)
{
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 jph_inverse_rotate(float4 q, float3 v)
{
    return jph_rotate(float4(-q.xyz, q.w), v);
}

float4 jph_rotation_step(float4 q, float3 angular_step)
{
    // Body::AddRotationStep uses an axis-angle left product, not quaternion
    // Euler integration, and skips steps with length <= 1e-6.
    float angle = length(angular_step);
    if (angle <= 1.0e-6f)
        return q;
    float half_angle = 0.5f * angle;
    float4 delta = float4(angular_step * (sin(half_angle) / angle), cos(half_angle));
    return normalize(jph_quat_mul(delta, q));
}

float3 jph_kinematic_angular_velocity(float4 target, float4 current, float requested_dt)
{
    float4 delta = jph_quat_mul(target, float4(-current.xyz, current.w));
    // q and -q choose the same shortest arc, including an exact PI tie.
    if (delta.w < 0.0f || (delta.w == 0.0f && (delta.x < 0.0f ||
        (delta.x == 0.0f && (delta.y < 0.0f || (delta.y == 0.0f && delta.z < 0.0f))))))
        delta = -delta;
    float length_squared = dot(delta.xyz, delta.xyz);
    if (length_squared < 1e-12f)
        return (2.0f * delta.xyz) / requested_dt;
    float imaginary_length = sqrt(length_squared);
    float angle = 2.0f * atan2(imaginary_length, delta.w);
    return (delta.xyz * (angle / imaginary_length)) / requested_dt;
}

float4 jph_kinematic_rotation_step(float4 rotation, float3 angular_step)
{
    float length_squared = dot(angular_step, angular_step);
    if (length_squared == 0.0f)
        return rotation;
    float sine_scale;
    float cosine;
    if (length_squared < 1e-12f)
    {
        // Continuous exponential at zero; a small target is not discarded
        // merely because collision substeps make each angular step tiny.
        sine_scale = 0.5f - length_squared / 48.0f;
        cosine = 1.0f - length_squared / 8.0f;
    }
    else
    {
        float angle = sqrt(length_squared);
        sine_scale = sin(0.5f * angle) / angle;
        cosine = cos(0.5f * angle);
    }
    return normalize(jph_quat_mul(float4(angular_step * sine_scale, cosine), rotation));
}

KinematicTarget jph_empty_kinematic_target()
{
    KinematicTarget target;
    target.positionSpeed = float4(0.0f);
    target.rotation = float4(0.0f);
    target.rootCOMDt = float4(0.0f);
    target.state = uint4(0u);
    return target;
}

float3 jph_clamp_speed(float3 velocity, float maximum)
{
    // Scale by a normal power of two before measuring the speed. Dividing by
    // the largest component can flush its reciprocal to zero on the GPU.
    float scale = max(abs(velocity.x), max(abs(velocity.y), abs(velocity.z)));
    if (scale == 0.0f)
        return velocity;
    // This also keeps the scaled cap finite: the remaining maximum is < 2*scale.
    if (scale <= maximum * 0.5f)
        return velocity;
    uint exponent = (asuint(scale) >> 23) & 255u;
    float normalization = asfloat(max(1u, 254u - exponent) << 23);
    float3 direction = velocity * normalization;
    float speed = length(direction);
    if (speed > maximum * normalization)
        velocity = (direction / speed) * maximum;
    return velocity;
}

bool jph_dynamic(BodyModel model)
{
    return (model.meta.z & ~MotionNotResident) == MotionDynamic;
}

bool jph_nonstatic(BodyModel model)
{
    return (model.meta.z & ~MotionNotResident) != MotionStatic;
}

bool jph_has_kinematic_move(KinematicTarget target)
{
    return (target.state.x & (TargetPending | TargetDriven)) != 0u;
}

BodyModel jph_effective_model(BodyModel author, KinematicTarget target)
{
    // This local view never writes BodyModels: accepted author Static remains
    // Static while the explicit GPU drive state supplies temporary movement.
    if ((author.meta.z & ~MotionNotResident) == MotionStatic &&
        (target.state.x & TargetTemporaryStatic) != 0u &&
        (target.state.x & TargetDriven) != 0u)
        author.meta.z = (author.meta.z & MotionNotResident) | MotionKinematic;
    return author;
}

bool jph_resident(BodyModel model)
{
    return (model.meta.z & MotionNotResident) == 0u;
}

bool jph_sensor(ShapeModel shape)
{
    return (shape.meta.w & ShapeSensor) != 0u;
}

bool jph_layers_collide(ShapeModel a, ShapeModel b)
{
    return (a.surface.y & (1u << b.surface.x)) != 0u &&
           (b.surface.y & (1u << a.surface.x)) != 0u;
}

float jph_inverse_mass(BodyModel model)
{
    return jph_dynamic(model) ? model.mass.x : 0.0f;
}

// The physical mass/inertia remain unprojected in BodyModel for force units.
float3 jph_allowed_axes(float locks)
{
    uint bits = uint(locks);
    return float3((bits & 1u) == 0u ? 1.0f : 0.0f,
                  (bits & 2u) == 0u ? 1.0f : 0.0f,
                  (bits & 4u) == 0u ? 1.0f : 0.0f);
}

float3 jph_project_linear(BodyModel model, float3 value)
{
    return value * jph_allowed_axes(model.mass.y);
}

float3 jph_project_angular(BodyModel model, BodyState state, float3 value)
{
    if (model.mass.z == 0.0f) return value;
    // MotionProperties::LockAngular constrains world components, independent
    // of the body's orientation and a recooked principal-inertia frame.
    return value * jph_allowed_axes(model.mass.z);
}

void jph_project_motion(BodyModel model, inout BodyState state)
{
    state.linearVelocity.xyz = jph_project_linear(model, state.linearVelocity.xyz);
    state.angularVelocity.xyz = jph_project_angular(model, state, state.angularVelocity.xyz);
}

// The linear sweep and the integrator share the same post-solver speeds
// and orientation. Position is left at the start of the translation.
void jph_prepare_drift(BodyModel model, inout BodyState state, float dt)
{
    jph_project_motion(model, state);
    if (jph_dynamic(model))
    {
        state.linearVelocity.xyz = jph_clamp_speed(state.linearVelocity.xyz, model.damping.z);
        state.angularVelocity.xyz = jph_clamp_speed(state.angularVelocity.xyz, model.damping.w);
        state.rotation = jph_rotation_step(state.rotation, state.angularVelocity.xyz * dt);
    }
    else
        state.rotation = jph_kinematic_rotation_step(state.rotation, state.angularVelocity.xyz * dt);
}

BodyState jph_sweep_state(SweepBody sweep)
{
    BodyState state;
    state.positionCOM = float4(sweep.positionCOM.xyz, 0.0f);
    state.rotation = sweep.rotation;
    state.linearVelocity = sweep.linearVelocity;
    state.angularVelocity = sweep.angularVelocity;
    return state;
}

float jph_sweep_role(BodyModel model, float3 displacement, float thresholdFactor)
{
    if (!jph_dynamic(model)) return 0.0f;
    float radius = abs(model.invInertiaDiagonal.w);
    float threshold = radius * thresholdFactor;
    return radius > 0.0f && dot(displacement, displacement) > threshold * threshold ? radius : -1.0f;
}

float3 jph_inverse_linear_mass(BodyModel model, float3 value)
{
    return jph_inverse_mass(model) * jph_project_linear(model, value);
}

float jph_inverse_mass_axis(BodyModel model, float3 axis)
{
    return dot(axis, jph_inverse_linear_mass(model, axis));
}

float3 jph_inverse_inertia(BodyModel model, BodyState state, float3 v)
{
    if (!jph_dynamic(model))
        return float3(0.0f);
    float4 inertia_to_world = jph_quat_mul(state.rotation, model.inertiaRotation);
    if (model.mass.z == 0.0f)
        return jph_rotate(inertia_to_world,
            model.invInertiaDiagonal.xyz * jph_inverse_rotate(inertia_to_world, v));
    // Match MotionProperties::MultiplyWorldSpaceInverseInertiaByVector:
    // mask world-space columns before R * diag(invI) * R^T, then its rows.
    float3 allowed = jph_allowed_axes(model.mass.z);
    return jph_rotate(inertia_to_world,
        model.invInertiaDiagonal.xyz * jph_inverse_rotate(inertia_to_world, v * allowed)) * allowed;
}

float3 jph_point_velocity(BodyModel model, BodyState state, float3 r)
{
    // Moving bodies provide prescribed velocity, but only Dynamic bodies
    // have nonzero inverse mass/inertia and receive solver impulses.
    return jph_nonstatic(model)
        ? state.linearVelocity.xyz + cross(state.angularVelocity.xyz, r)
        : float3(0.0f);
}

void jph_apply_gyroscopic_force(BodyModel model, inout BodyState state, float dt)
{
    // MotionProperties::ApplyGyroscopicForceInternal also runs with partial
    // world DOF locks; they do not remove gyroscopic coupling in the allowed axes.
    float4 inertia_to_world = jph_quat_mul(state.rotation, model.inertiaRotation);
    float3 local_velocity = jph_inverse_rotate(inertia_to_world, state.angularVelocity.xyz);
    float3 local_momentum = local_velocity / model.invInertiaDiagonal.xyz;
    float3 new_momentum = local_momentum - dt * cross(local_velocity, local_momentum);
    float new_length_sq = dot(new_momentum, new_momentum);
    new_momentum = new_length_sq > 0.0f
        ? new_momentum * sqrt(dot(local_momentum, local_momentum) / new_length_sq)
        : float3(0.0f);
    // Apply the world-axis locks before contacts consume the new angular velocity.
    state.angularVelocity.xyz = jph_project_angular(
        model, state, jph_rotate(inertia_to_world, model.invInertiaDiagonal.xyz * new_momentum));
}

float3 jph_acceleration(BodyModel model, BodyForces forces, float3 gravity)
{
    return jph_dynamic(model) ? jph_project_linear(model, gravity * model.mass.w) + jph_inverse_linear_mass(model, forces.force.xyz) : float3(0.0f);
}

void jph_apply_body_inputs(BodyModel model, inout BodyState state, BodyForces forces, float4 gravity_dt)
{
    float dt = gravity_dt.w;
    jph_project_motion(model, state);
    jph_apply_gyroscopic_force(model, state, dt);
    state.linearVelocity.xyz += dt * jph_acceleration(model, forces, gravity_dt.xyz);
    state.angularVelocity.xyz += dt * jph_inverse_inertia(model, state, forces.torque.xyz);
    state.linearVelocity.xyz *= max(0.0f, 1.0f - model.damping.x * dt);
    state.angularVelocity.xyz *= max(0.0f, 1.0f - model.damping.y * dt);
    state.linearVelocity.xyz = jph_clamp_speed(state.linearVelocity.xyz, model.damping.z);
    state.angularVelocity.xyz = jph_clamp_speed(state.angularVelocity.xyz, model.damping.w);
}

void jph_tangents(float3 normal, out float3 t1, out float3 t2)
{
    // Vec3::GetNormalizedPerpendicular and ContactConstraint::GetTangents.
    t1 = abs(normal.x) > abs(normal.y)
        ? normalize(float3(normal.z, 0.0f, -normal.x))
        : normalize(float3(0.0f, normal.z, -normal.y));
    t2 = cross(normal, t1);
}

float jph_effective_mass(BodyModel ma, BodyState sa, float3 ra,
                         BodyModel mb, BodyState sb, float3 rb, float3 axis)
{
    float3 ja = cross(ra, axis);
    float3 jb = cross(rb, axis);
    float inverse_effective_mass = jph_inverse_mass_axis(ma, axis) + jph_inverse_mass_axis(mb, axis)
        + dot(ja, jph_inverse_inertia(ma, sa, ja))
        + dot(jb, jph_inverse_inertia(mb, sb, jb));
    return inverse_effective_mass > 0.0f ? 1.0f / inverse_effective_mass : 0.0f;
}

float jph_jv(BodyModel ma, BodyState sa, float3 ra,
             BodyModel mb, BodyState sb, float3 rb, float3 axis)
{
    // ContactConstraintPart::SolveVelocityConstraintGetTotalLambda convention.
    return dot(axis, jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb));
}

void jph_apply_impulse(BodyModel ma, inout BodyState sa, float3 ra,
                       BodyModel mb, inout BodyState sb, float3 rb,
                       float3 axis, float delta_lambda)
{
    if (jph_dynamic(ma))
    {
        sa.linearVelocity.xyz -= delta_lambda * jph_inverse_linear_mass(ma, axis);
        sa.angularVelocity.xyz -= delta_lambda * jph_inverse_inertia(ma, sa, cross(ra, axis));
    }
    if (jph_dynamic(mb))
    {
        sb.linearVelocity.xyz += delta_lambda * jph_inverse_linear_mass(mb, axis);
        sb.angularVelocity.xyz += delta_lambda * jph_inverse_inertia(mb, sb, cross(rb, axis));
    }
}

// Return the impulse applied to B, and keep the incoming point velocity for reports.
float3 jph_solve_sweep_impact(BodyModel ma, inout BodyState sa, float3 ra,
                              BodyModel mb, inout BodyState sb, float3 rb, float3 normal,
                              float friction, float restitution, float minRestitutionSpeed,
                              out float3 incoming)
{
    incoming = jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb);
    float speed = max(0.0f, dot(incoming, normal));
    float bounce = speed > minRestitutionSpeed ? restitution : 0.0f;
    float lambda = (1.0f + bounce) * speed * jph_effective_mass(ma, sa, ra, mb, sb, rb, normal);
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, normal, lambda);
    float3 impulse = normal * lambda;
    if (friction > 0.0f)
    {
        float3 relative = jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb);
        float3 tangent = relative - normal * dot(normal, relative);
        float tangentSpeed = length(tangent);
        if (tangentSpeed > 0.0f)
        {
            tangent /= tangentSpeed;
            float tangentLambda = min(friction * lambda,
                tangentSpeed * jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent));
            jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent, tangentLambda);
            impulse += tangent * tangentLambda;
        }
    }
    return impulse;
}

void jph_contact_points(Contact contact, BodyState sa, BodyState sb,
                        out float3 pa, out float3 pb, out float3 ra, out float3 rb)
{
    pa = sa.positionCOM.xyz + jph_rotate(sa.rotation, contact.localAnchorA.xyz);
    pb = sb.positionCOM.xyz + jph_rotate(sb.rotation, contact.localAnchorB.xyz);
    float3 midpoint = 0.5f * (pa + pb);
    ra = midpoint - sa.positionCOM.xyz;
    rb = midpoint - sb.positionCOM.xyz;
}

float jph_capsule_max_abs(float3 value)
{
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}

float jph_capsule_length(float3 value)
{
    float scale = jph_capsule_max_abs(value);
    return scale > 0.0f ? scale * length(value / scale) : 0.0f;
}

float3 jph_capsule_unit(float3 value)
{
    float3 scaled = value / jph_capsule_max_abs(value);
    return scaled / length(scaled);
}

void jph_capsule_segment_closest(float3 a0, float3 a1, float3 b0, float3 b1,
                                 out float3 pa, out float3 pb)
{
    float3 da = a1 - a0;
    float3 db = b1 - b0;
    float scale_a = jph_capsule_max_abs(da);
    float scale_b = jph_capsule_max_abs(db);
    float3 ua = scale_a > 0.0f ? da / scale_a : float3(0.0f);
    float3 ub = scale_b > 0.0f ? db / scale_b : float3(0.0f);
    float aa = dot(ua, ua);
    float bb = dot(ub, ub);
    float3 delta = a0 - b0;
    float s = 0.0f;
    float t = 0.0f;
    if (scale_a == 0.0f)
    {
        if (scale_b > 0.0f)
            t = clamp(dot(ub, delta) / bb, 0.0f, scale_b);
    }
    else if (scale_b == 0.0f)
        s = clamp(-dot(ua, delta) / aa, 0.0f, scale_a);
    else
    {
        float ab = dot(ua, ub);
        float ad = dot(ua, delta);
        float bd = dot(ub, delta);
        float3 perpendicular = cross(ua, ub);
        float cross_scale = jph_capsule_max_abs(perpendicular);
        if (cross_scale > 0.0f)
        {
            float3 scaled_cross = perpendicular / cross_scale;
            // Dividing out the nonzero cross scale before squaring retains
            // long, thin, nearly parallel segments that actually intersect.
            s = clamp(dot(cross(ub, delta), scaled_cross) /
                      (cross_scale * dot(scaled_cross, scaled_cross)), 0.0f, scale_a);
        }
        t = (ab * s + bd) / bb;
        if (t < 0.0f)
        {
            t = 0.0f;
            s = clamp(-ad / aa, 0.0f, scale_a);
        }
        else if (t > scale_b)
        {
            t = scale_b;
            s = clamp((ab * t - ad) / aa, 0.0f, scale_a);
        }
    }
    pa = a0 + s * ua;
    pb = b0 + t * ub;
}

// Geometric primary, independent of speculative admission and manifold reduction.
// status=0 unsupported, 1 available, 2 nonfinite; every field is initialized.
// Resolved separation > 0 is disjoint, 0 touching, < 0 penetrating.
// The unit normal points A -> B; dot(pointA-pointB, normal) = -separation.
struct JphPairMetric
{
    float3 normal;
    float separation;
    float3 pointA;
    uint feature;
    float3 pointB;
    uint status;
};

JphPairMetric jph_empty_pair_metric()
{
    JphPairMetric metric;
    metric.normal = float3(1.0f, 0.0f, 0.0f);
    metric.separation = 0.0f;
    metric.pointA = float3(0.0f);
    metric.pointB = float3(0.0f);
    metric.feature = 0u;
    metric.status = 0u;
    return metric;
}

JphPairMetric jph_flip_pair_metric(JphPairMetric metric)
{
    metric.normal = -metric.normal;
    float3 pointA = metric.pointA;
    metric.pointA = metric.pointB;
    metric.pointB = pointA;
    return metric;
}

void jph_sphere_sphere(float3 ca, float radius_a, float3 cb, float radius_b,
                       out float3 normal, out float penetration,
                       out float3 pa, out float3 pb)
{
    float3 delta = cb - ca;
    float distance = jph_capsule_length(delta);
    // Only an exactly coincident center pair needs a deterministic tie normal.
    normal = jph_capsule_max_abs(delta) > 0.0f ? jph_capsule_unit(delta) : float3(1.0f, 0.0f, 0.0f);
    penetration = radius_a + radius_b - distance;
    pa = ca + radius_a * normal;
    pb = cb - radius_b * normal;
}

void jph_sphere_box(float3 sphere_center, float sphere_radius,
                    float3 box_center, float4 box_rotation, float4 box_geometry,
                    out float3 normal_sphere_to_box, out float penetration,
                    out float3 point_sphere, out float3 point_box, out uint feature)
{
    // BoxShape::ExcludeConvexRadius exposes the reduced core plus a Minkowski
    // convex radius. Thus sphere/core-box distance uses the sum of the radii.
    // This retains the rounded edge/corner geometry instead of ignoring the
    // default box convex radius. Deep/degenerate contacts are handled explicitly.
    float convex_radius = box_geometry.w;
    float3 core_half_extent = max(box_geometry.xyz - convex_radius, float3(0.0f));
    float3 center = jph_inverse_rotate(box_rotation, sphere_center - box_center);
    float3 closest_core = clamp(center, -core_half_extent, core_half_extent);
    float3 delta = center - closest_core;
    float3 outward;
    float3 box_point_local;
    if (jph_capsule_max_abs(delta) > 0.0f)
    {
        float distance = jph_capsule_length(delta);
        outward = jph_capsule_unit(delta);
        penetration = sphere_radius + convex_radius - distance;
        box_point_local = closest_core + convex_radius * outward;
        // Outside-core contacts include a face, edge or corner feature.
        feature = 8u + (center.x < -core_half_extent.x ? 1u : center.x > core_half_extent.x ? 2u : 0u)
            + 3u * (center.y < -core_half_extent.y ? 1u : center.y > core_half_extent.y ? 2u : 0u)
            + 9u * (center.z < -core_half_extent.z ? 1u : center.z > core_half_extent.z ? 2u : 0u);
    }
    else
    {
        // An embedded center must leave through its nearest core face. A
        // closest-point clamp alone gives zero and would miss this collision.
        float3 face_distance = core_half_extent - abs(center);
        uint axis = 0u;
        if (face_distance.y < face_distance.x)
            axis = 1u;
        if (face_distance.z < face_distance[axis])
            axis = 2u;
        outward = float3(0.0f);
        outward[axis] = center[axis] >= 0.0f ? 1.0f : -1.0f;
        float distance_to_face = max(face_distance[axis], 0.0f);
        penetration = sphere_radius + convex_radius + distance_to_face;
        box_point_local = center + (distance_to_face + convex_radius) * outward;
        feature = 2u * axis + (center[axis] >= 0.0f ? 1u : 0u);
    }
    float3 outward_world = jph_rotate(box_rotation, outward);
    normal_sphere_to_box = -outward_world;
    point_sphere = sphere_center - sphere_radius * outward_world;
    point_box = box_center + jph_rotate(box_rotation, box_point_local);
}


// Rounded boxes are their reduced OBB cores plus a spherical convex radius.
// All work below is bounded: 15 SAT axes and 16 vertex/OBB + 144 edge/edge
// candidates. SAT separation alone is not a rounded-box distance query.
struct JphCoreBox
{
    float3 center;
    float3 axis[3];
    float3 halfExtent;
    float radius;
};

struct JphBoxPoint
{
    float3 pointA;
    float3 pointB;
    float depth;
    uint feature;
};

struct JphBoxManifold
{
    float3 normal;
    uint count;
    // Exact primary distance test, independent of contact-area point reduction.
    // Patch overlap must not be inferred from the speculative contact band.
    float penetration;
    JphBoxPoint points[MaxContactPointsPerPair];
};

JphCoreBox jph_core_box(float3 center, float4 rotation, float4 geometry)
{
    JphCoreBox box;
    box.center = center;
    box.axis[0] = jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f));
    box.axis[1] = jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f));
    box.axis[2] = jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f));
    box.halfExtent = max(geometry.xyz - geometry.w, float3(0.0f));
    box.radius = geometry.w;
    return box;
}

float3 jph_core_vertex(JphCoreBox box, uint index)
{
    return box.center
        + box.axis[0] * ((index & 1u) != 0u ? box.halfExtent.x : -box.halfExtent.x)
        + box.axis[1] * ((index & 2u) != 0u ? box.halfExtent.y : -box.halfExtent.y)
        + box.axis[2] * ((index & 4u) != 0u ? box.halfExtent.z : -box.halfExtent.z);
}

float3 jph_core_closest(JphCoreBox box, float3 point)
{
    float3 delta = point - box.center;
    float3 local = float3(dot(delta, box.axis[0]), dot(delta, box.axis[1]), dot(delta, box.axis[2]));
    local = clamp(local, -box.halfExtent, box.halfExtent);
    return box.center + box.axis[0] * local.x + box.axis[1] * local.y + box.axis[2] * local.z;
}

void jph_core_edge(JphCoreBox box, uint index, out float3 begin, out float3 end)
{
    uint axis = index / 4u;
    uint side = index % 4u;
    uint u = (axis + 1u) % 3u;
    uint v = (axis + 2u) % 3u;
    float3 center = box.center
        + box.axis[u] * ((side & 1u) != 0u ? box.halfExtent[u] : -box.halfExtent[u])
        + box.axis[v] * ((side & 2u) != 0u ? box.halfExtent[v] : -box.halfExtent[v]);
    begin = center - box.axis[axis] * box.halfExtent[axis];
    end = center + box.axis[axis] * box.halfExtent[axis];
}

void jph_segment_closest(float3 a0, float3 a1, float3 b0, float3 b1,
                          out float3 pointA, out float3 pointB)
{
    // A nonzero edge/cross product is not collapsed by a world-unit or angle
    // epsilon. Share the scale-aware finite-segment primary used by capsules.
    jph_capsule_segment_closest(a0, a1, b0, b1, pointA, pointB);
}

void jph_closest_candidate(float3 a, float3 b, uint id, inout float best,
                            inout float3 pointA, inout float3 pointB, inout uint feature)
{
    float3 delta = b - a;
    float distance = jph_capsule_length(delta);
    if (id == 0u || distance < best)
    {
        best = distance;
        pointA = a;
        pointB = b;
        feature = id;
    }
}

void jph_core_distance(JphCoreBox a, JphCoreBox b, out float3 pointA, out float3 pointB,
                        out float distance, out uint feature)
{
    distance = 3.402823466e+38f;
    pointA = a.center;
    pointB = b.center;
    feature = 0u;
    for (uint vertex = 0u; vertex < 8u; ++vertex)
    {
        float3 va = jph_core_vertex(a, vertex);
        float3 vb = jph_core_vertex(b, vertex);
        jph_closest_candidate(va, jph_core_closest(b, va), vertex,
                              distance, pointA, pointB, feature);
        jph_closest_candidate(jph_core_closest(a, vb), vb, 8u + vertex,
                              distance, pointA, pointB, feature);
    }
    for (uint edgeA = 0u; edgeA < 12u; ++edgeA)
    {
        float3 a0, a1;
        jph_core_edge(a, edgeA, a0, a1);
        for (uint edgeB = 0u; edgeB < 12u; ++edgeB)
        {
            float3 b0, b1, pa, pb;
            jph_core_edge(b, edgeB, b0, b1);
            jph_segment_closest(a0, a1, b0, b1, pa, pb);
            jph_closest_candidate(pa, pb, 16u + edgeA * 12u + edgeB,
                                  distance, pointA, pointB, feature);
        }
    }
}

float jph_core_axis_radius(JphCoreBox box, float3 n)
{
    return abs(dot(box.axis[0], n)) * box.halfExtent.x + abs(dot(box.axis[1], n)) * box.halfExtent.y
         + abs(dot(box.axis[2], n)) * box.halfExtent.z;
}

bool jph_core_sat(JphCoreBox a, JphCoreBox b, out float3 normal, out float depth, out uint feature)
{
    float3 delta = b.center - a.center;
    depth = 3.402823466e+38f;
    normal = a.axis[0];
    feature = 0u;
    bool overlapping = true;
    for (uint axis = 0u; axis < 15u; ++axis)
    {
        float3 n;
        if (axis < 3u)
            n = a.axis[axis];
        else if (axis < 6u)
            n = b.axis[axis - 3u];
        else
            n = cross(a.axis[(axis - 6u) / 3u], b.axis[(axis - 6u) % 3u]);
        float axis_scale = jph_capsule_max_abs(n);
        if (axis_scale == 0.0f)
            continue; // Only exactly parallel edges omit this facet class.
        n /= axis_scale;
        n /= length(n);
        float radiusA = jph_core_axis_radius(a, n);
        float radiusB = jph_core_axis_radius(b, n);
        float center_distance = dot(delta, n);
        float penetration = radiusA + radiusB - abs(center_distance);
        overlapping = overlapping && penetration >= 0.0f;
        // Face axes precede edge axes; exact ties retain that stable order.
        if (penetration < depth)
        {
            depth = penetration;
            normal = center_distance >= 0.0f ? n : -n;
            feature = axis;
        }
    }
    return overlapping;
}

uint jph_clip_box_face(float3 input_vertices[8], uint count, float3 normal, float limit,
                        out float3 output_vertices[8])
{
    // The caller supplies a nonempty polygon. Only the returned vertex range
    // is initialized; unused array tails are not geometry.
    uint output_count = 0u;
    float3 previous = input_vertices[count - 1u];
    float previous_distance = limit - dot(normal, previous);
    for (uint index = 0u; index < 8u; ++index)
    {
        if (index >= count)
            break;
        float3 current = input_vertices[index];
        float current_distance = limit - dot(normal, current);
        bool current_inside = current_distance >= 0.0f;
        // A boundary endpoint is emitted by the vertex branch exactly once.
        // Only a strict sign crossing creates a new polygon vertex.
        if ((previous_distance < 0.0f && current_distance > 0.0f)
            || (previous_distance > 0.0f && current_distance < 0.0f))
        {
            float fraction = previous_distance / (previous_distance - current_distance);
            output_vertices[output_count++] = previous + fraction * (current - previous);
        }
        if (current_inside)
            output_vertices[output_count++] = current;
        previous = current;
        previous_distance = current_distance;
    }
    return output_count;
}

uint jph_box_face_points(JphCoreBox a, JphCoreBox b, float3 normal, float speculative_distance,
                          inout JphBoxPoint points[9], uint count)
{
    // Extra contacts use real flat patches of the rounded shapes. Their plane
    // is at outer half extent, but their tangent extents belong to the CORE.
    // Full outer-box face quads would turn rounded corners into sharp corners.
    uint reference_shape = 0u;
    uint reference_axis = 0u;
    float alignment = 3.402823466e+38f;
    for (uint shape = 0u; shape < 2u; ++shape)
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            float3 direction = shape == 0u ? a.axis[axis] : b.axis[axis];
            float3 perpendicular = cross(direction, normal);
            float error = dot(perpendicular, perpendicular);
            if (error < alignment)
            {
                alignment = error;
                reference_shape = shape;
                reference_axis = axis;
            }
        }
    JphCoreBox reference = a;
    JphCoreBox incident = b;
    if (reference_shape != 0u)
    {
        reference = b;
        incident = a;
    }
    float3 outward = reference_shape == 0u ? normal : -normal;
    float3 reference_normal = dot(reference.axis[reference_axis], outward) >= 0.0f
        ? reference.axis[reference_axis] : -reference.axis[reference_axis];
    float3 plane_origin = reference.center
        + reference_normal * (reference.halfExtent[reference_axis] + reference.radius);
    uint u = (reference_axis + 1u) % 3u;
    uint v = (reference_axis + 2u) % 3u;
    uint incident_axis = 0u;
    if (abs(dot(incident.axis[1], outward)) > abs(dot(incident.axis[incident_axis], outward)))
        incident_axis = 1u;
    if (abs(dot(incident.axis[2], outward)) > abs(dot(incident.axis[incident_axis], outward)))
        incident_axis = 2u;
    float3 incident_normal = dot(incident.axis[incident_axis], outward) >= 0.0f
        ? -incident.axis[incident_axis] : incident.axis[incident_axis];
    float3 face_origin = incident.center
        + incident_normal * (incident.halfExtent[incident_axis] + incident.radius);
    uint iu = (incident_axis + 1u) % 3u;
    uint iv = (incident_axis + 2u) % 3u;
    float3 du = incident.axis[iu] * incident.halfExtent[iu];
    float3 dv = incident.axis[iv] * incident.halfExtent[iv];
    float3 polygon[8];
    polygon[0] = face_origin - du - dv;
    polygon[1] = face_origin + du - dv;
    polygon[2] = face_origin + du + dv;
    polygon[3] = face_origin - du + dv;
    // Map the actual incident flat patch onto the reference plane along the
    // common contact direction BEFORE clipping. A slightly tilted normal must
    // not collapse a resting patch to one point, nor move a clipped endpoint
    // outside the reference patch. The most aligned reference/incident axes
    // guarantee nonzero plane denominators (absolute dot >= 1/sqrt(3)).
    float reference_denominator = dot(reference_normal, outward);
    for (uint vertex = 0u; vertex < 4u; ++vertex)
    {
        float distance = dot(plane_origin - polygon[vertex], reference_normal) / reference_denominator;
        polygon[vertex] += distance * outward;
    }
    uint polygon_count = 4u;
    // A quad clipped by four rectangle planes has at most eight vertices.
    for (uint side = 0u; side < 4u; ++side)
    {
        if (polygon_count == 0u)
            break;
        uint tangent = side < 2u ? u : v;
        float3 side_normal = (side & 1u) == 0u ? reference.axis[tangent] : -reference.axis[tangent];
        float limit = dot(side_normal, reference.center) + reference.halfExtent[tangent];
        float3 clipped[8];
        polygon_count = jph_clip_box_face(polygon, polygon_count, side_normal, limit, clipped);
        for (uint vertex = 0u; vertex < 8u; ++vertex)
            if (vertex < polygon_count)
                polygon[vertex] = clipped[vertex];
    }
    float incident_denominator = dot(incident_normal, outward);
    for (uint vertex = 0u; vertex < 8u; ++vertex)
    {
        if (vertex >= polygon_count)
            break;
        float3 reference_point = polygon[vertex];
        // Invert the projection to recover a point on the ORIGINAL incident
        // flat patch. Clipping in the reference plane preserves both finite
        // patch domains because this plane-to-plane map is affine/invertible.
        float distance = dot(face_origin - reference_point, incident_normal) / incident_denominator;
        float penetration = -distance;
        if (penetration < -speculative_distance)
            continue;
        float3 incident_point = reference_point + distance * outward;
        JphBoxPoint point;
        point.pointA = reference_shape == 0u ? reference_point : incident_point;
        point.pointB = reference_shape == 0u ? incident_point : reference_point;
        point.depth = penetration;
        point.feature = 1024u + reference_shape * 24u + reference_axis * 8u + vertex;
        points[count++] = point;
    }
    return count;
}

// A face normal gives the exact Euclidean separation when the incident support
// vertex projects inside that finite face. Keep that normal analytically:
// subtracting two reconstructed witnesses can turn a sub-ulp tangential error
// into most of the normal of an almost touching, sliding pair.
bool jph_separated_box_face(JphCoreBox reference, JphCoreBox incident, out float3 outward,
                            out float gap, out float3 on_reference, out float3 on_incident, out uint vertex)
{
    float3 delta = incident.center - reference.center;
    for (uint i = 0u; i < 3u; ++i)
    {
        float center_distance = dot(delta, reference.axis[i]);
        float3 n = center_distance >= 0.0f ? reference.axis[i] : -reference.axis[i];
        float3 projection = float3(dot(incident.axis[0], n), dot(incident.axis[1], n), dot(incident.axis[2], n));
        float distance = abs(center_distance) - (reference.halfExtent[i] + dot(abs(projection), incident.halfExtent));
        if (distance < 0.0f)
            continue;
        uint v = (projection.x < 0.0f ? 1u : 0u) | (projection.y < 0.0f ? 2u : 0u) | (projection.z < 0.0f ? 4u : 0u);
        float3 p = jph_core_vertex(incident, v);
        float3 offset = p - reference.center;
        uint u = (i + 1u) % 3u, t = (i + 2u) % 3u;
        if (abs(dot(offset, reference.axis[u])) <= reference.halfExtent[u] &&
            abs(dot(offset, reference.axis[t])) <= reference.halfExtent[t])
        {
            outward = n;
            gap = distance;
            on_reference = p - n * distance;
            on_incident = p;
            vertex = v;
            return true;
        }
    }
    outward = on_reference = on_incident = float3(0.0f);
    gap = 0.0f;
    vertex = 0u;
    return false;
}

JphPairMetric jph_core_box_pair_metric(JphCoreBox a, JphCoreBox b)
{
    float3 normal;
    float core_depth;
    uint axis_feature;
    bool core_overlap = jph_core_sat(a, b, normal, core_depth, axis_feature);
    float3 pa, pb, face_normal;
    float core_distance;
    uint distance_feature;
    float penetration;
    if (core_overlap)
    {
        // At the minimum translation, the two core boundaries touch. Their
        // exact closest feature pair supplies matching support witnesses,
        // including partial face overlap, edge/edge and contained boxes.
        JphCoreBox translated = b;
        translated.center += core_depth * normal;
        jph_core_distance(a, translated, pa, pb, core_distance, distance_feature);
        pb -= core_depth * normal;
        penetration = core_depth + a.radius + b.radius;
    }
    else if (jph_separated_box_face(a, b, face_normal, core_distance, pa, pb, distance_feature))
    {
        normal = face_normal;
        distance_feature += 8u; // vertex of B against A
        penetration = a.radius + b.radius - core_distance;
    }
    else if (jph_separated_box_face(b, a, face_normal, core_distance, pb, pa, distance_feature))
    {
        normal = -face_normal;
        penetration = a.radius + b.radius - core_distance;
    }
    else
    {
        jph_core_distance(a, b, pa, pb, core_distance, distance_feature);
        float3 witness = pb - pa;
        if (core_distance > 0.0f)
        {
            float3 n = jph_capsule_unit(witness);
            // A closest-pair normal must separate the *whole* boxes. At touching
            // precision the residual can point along a common face instead.
            float projected_gap = dot(b.center - a.center, n) - (jph_core_axis_radius(a, n) + jph_core_axis_radius(b, n));
            if (projected_gap > 0.0f)
                normal = n;
            else
                core_distance = -core_depth; // SAT already certified this separating axis.
        }
        penetration = a.radius + b.radius - core_distance;
    }

    JphPairMetric metric;
    metric.normal = normal;
    metric.separation = -penetration;
    metric.pointA = pa + a.radius * normal;
    metric.pointB = pb - b.radius * normal;
    metric.feature = core_overlap ? 64u + axis_feature : 128u + distance_feature;
    metric.status = 1u;
    return metric;
}

JphPairMetric jph_box_box_metric(float3 centerA, float4 rotationA, float4 geometryA,
                                float3 centerB, float4 rotationB, float4 geometryB)
{
    JphCoreBox a = jph_core_box(float3(0.0f), rotationA, geometryA);
    JphCoreBox b = jph_core_box(centerB - centerA, rotationB, geometryB);
    JphPairMetric metric = jph_core_box_pair_metric(a, b);
    metric.pointA += centerA;
    metric.pointB += centerA;
    return metric;
}

JphBoxManifold jph_box_box(float3 centerA, float4 rotationA, float4 geometryA,
                            float3 centerB, float4 rotationB, float4 geometryB, float speculative_distance)
{
    // Pair-relative coordinates keep the narrow phase independent of a large
    // shared world translation. Anchors are converted to world only at output.
    JphCoreBox a = jph_core_box(float3(0.0f), rotationA, geometryA);
    JphCoreBox b = jph_core_box(centerB - centerA, rotationB, geometryB);
    JphPairMetric primary = jph_core_box_pair_metric(a, b);
    float3 normal = primary.normal;
    float penetration = -primary.separation;

    JphBoxManifold manifold;
    manifold.normal = normal;
    manifold.count = 0u;
    manifold.penetration = penetration;
    if (penetration < -speculative_distance)
        return manifold;
    JphBoxPoint candidates[9];
    candidates[0].pointA = primary.pointA;
    candidates[0].pointB = primary.pointB;
    candidates[0].depth = dot(candidates[0].pointA - candidates[0].pointB, normal);
    candidates[0].feature = primary.feature;
    uint count = jph_box_face_points(a, b, normal, speculative_distance, candidates, 1u);

    // Remove duplicate boundary points before selecting a support polygon.
    float extent = max(1.0f, max(max(geometryA.x, geometryA.y), max(geometryA.z,
                              max(max(geometryB.x, geometryB.y), geometryB.z))));
    float tolerance_squared = 1e-12f * extent * extent;
    uint unique_count = 0u;
    for (uint point = 0u; point < 9u; ++point)
    {
        if (point >= count)
            break;
        bool duplicate = false;
        for (uint other = 0u; other < 9u; ++other)
        {
            if (other >= unique_count)
                break;
            float3 da = candidates[point].pointA - candidates[other].pointA;
            float3 db = candidates[point].pointB - candidates[other].pointB;
            if (dot(da, da) + dot(db, db) <= tolerance_squared)
            {
                if (candidates[point].depth > candidates[other].depth)
                    candidates[other] = candidates[point];
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            candidates[unique_count++] = candidates[point];
    }

    // Depth first; equal-depth contacts prefer distance from the patch center.
    // Then retain the longest segment and its extreme points on both sides,
    // following Jolt's tangent-plane area reduction to at most four points.
    float3 center = float3(0.0f);
    for (uint point = 0u; point < 9u; ++point)
        if (point < unique_count)
            center += 0.5f * (candidates[point].pointA + candidates[point].pointB);
    center /= float(unique_count);
    uint first = 0u;
    float first_distance = -1.0f;
    float depth_tolerance = 1e-6f * extent;
    for (uint point = 0u; point < 9u; ++point)
    {
        if (point >= unique_count)
            break;
        float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - center;
        delta -= normal * dot(delta, normal);
        float distance = dot(delta, delta);
        if (candidates[point].depth > candidates[first].depth + depth_tolerance ||
            (abs(candidates[point].depth - candidates[first].depth) <= depth_tolerance && distance > first_distance))
        {
            first = point;
            first_distance = distance;
        }
    }
    uint selected[4];
    selected[0] = first;
    uint selected_count = 1u;
    if (unique_count > 1u)
    {
        uint second = first;
        float longest = -1.0f;
        float3 p1 = 0.5f * (candidates[first].pointA + candidates[first].pointB);
        for (uint point = 0u; point < 9u; ++point)
        {
            if (point >= unique_count)
                break;
            if (point == first)
                continue;
            float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - p1;
            delta -= normal * dot(delta, normal);
            float distance = dot(delta, delta);
            if (distance > longest)
            {
                longest = distance;
                second = point;
            }
        }
        selected[selected_count++] = second;
        float3 segment = 0.5f * (candidates[second].pointA + candidates[second].pointB) - p1;
        float3 perpendicular = cross(segment, normal);
        float minimum = 0.0f;
        float maximum = 0.0f;
        uint negative = 0xffffffffu;
        uint positive = 0xffffffffu;
        for (uint point = 0u; point < 9u; ++point)
        {
            if (point >= unique_count)
                break;
            if (point == first || point == second)
                continue;
            float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - p1;
            float side = dot(perpendicular, delta);
            if (side < minimum)
            {
                minimum = side;
                negative = point;
            }
            else if (side > maximum)
            {
                maximum = side;
                positive = point;
            }
        }
        if (negative != 0xffffffffu)
            selected[selected_count++] = negative;
        if (positive != 0xffffffffu)
            selected[selected_count++] = positive;
    }
    manifold.count = selected_count;
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < selected_count)
        {
            JphBoxPoint contact = candidates[selected[point]];
            contact.pointA += centerA;
            contact.pointB += centerA;
            manifold.points[point] = contact;
        }
    return manifold;
}

// Capsule geometry is a finite Y-axis segment plus its actual spherical radius.
// The helpers below use pair-relative, scaled coordinates; no sampling, angular
// parallel threshold or enlarged sharp-box approximation decides a contact.
bool jph_capsule_clip_parameter(float constant, float slope, inout float low, inout float high)
{
    if (slope > 0.0f)
        low = max(low, -constant / slope);
    else if (slope < 0.0f)
        high = min(high, -constant / slope);
    else if (constant < 0.0f)
        return false;
    return low <= high;
}

bool jph_capsule_core_box_intersects(float3 a, float3 b, float3 extent)
{
    float low = 0.0f;
    float high = 1.0f;
    float3 direction = b - a;
    for (uint axis = 0u; axis < 3u; ++axis)
        if (!jph_capsule_clip_parameter(extent[axis] + a[axis], direction[axis], low, high) ||
            !jph_capsule_clip_parameter(extent[axis] - a[axis], -direction[axis], low, high))
            return false;
    return true;
}

void jph_capsule_segment_box_closest(float3 a, float3 b, float3 extent,
                                     out float3 p, out float3 q)
{
    float3 direction = b - a;
    float breaks[8];
    breaks[0] = 0.0f;
    breaks[1] = 1.0f;
    uint count = 2u;
    for (uint axis = 0u; axis < 3u; ++axis)
        if (direction[axis] != 0.0f)
            for (uint side = 0u; side < 2u; ++side)
            {
                float plane = side == 0u ? -extent[axis] : extent[axis];
                float t = (plane - a[axis]) / direction[axis];
                if (t > 0.0f && t < 1.0f)
                    breaks[count++] = t;
            }
    for (uint index = 1u; index < count; ++index)
    {
        float value = breaks[index];
        uint position = index;
        for (uint move = 0u; move < 7u && position > 0u; ++move)
        {
            if (breaks[position - 1u] <= value)
                break;
            breaks[position] = breaks[position - 1u];
            --position;
        }
        breaks[position] = value;
    }
    p = a;
    q = clamp(a, -extent, extent);
    float best = jph_capsule_length(p - q);
    float3 end_q = clamp(b, -extent, extent);
    float end_distance = jph_capsule_length(b - end_q);
    if (end_distance < best)
    {
        p = b;
        q = end_q;
        best = end_distance;
    }
    // Distance squared is quadratic on each fixed outside-axis interval.
    // At most six slab crossings produce seven nonempty intervals.
    for (uint interval = 0u; interval + 1u < count; ++interval)
    {
        float start = breaks[interval];
        float width = breaks[interval + 1u] - start;
        if (width <= 0.0f)
            continue;
        float3 middle = a + (start + 0.5f * width) * direction;
        float3 at_start = a + start * direction;
        float quadratic = 0.0f;
        float linear = 0.0f;
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            float side = middle[axis] < -extent[axis] ? -1.0f :
                         middle[axis] > extent[axis] ? 1.0f : 0.0f;
            if (side == 0.0f)
                continue;
            quadratic += direction[axis] * direction[axis];
            linear += (at_start[axis] - side * extent[axis]) * direction[axis];
        }
        float offset = quadratic > 0.0f ? clamp(-linear / quadratic, 0.0f, width) : 0.0f;
        float3 candidate = at_start + offset * direction;
        float3 closest = clamp(candidate, -extent, extent);
        float distance = jph_capsule_length(candidate - closest);
        if (distance < best)
        {
            best = distance;
            p = candidate;
            q = closest;
        }
    }
}

void jph_capsule_core_box_mtd(float3 center, float3 axis, float half_height, float3 extent,
                             out float3 normal, out float depth, out uint feature)
{
    depth = 3.402823466e+38f;
    normal = float3(1.0f, 0.0f, 0.0f);
    feature = 0u;
    // The box-minus-segment zonotope has four generators. Their pairwise
    // cross products give exactly these six facet-normal classes. For an
    // intersecting core its nearest facet slack is the true exit distance.
    for (uint candidate = 0u; candidate < 6u; ++candidate)
    {
        float3 basis = float3(0.0f);
        basis[candidate % 3u] = 1.0f;
        float3 n = candidate < 3u ? basis : cross(axis, basis);
        if (jph_capsule_max_abs(n) == 0.0f)
            continue;
        n = jph_capsule_unit(n);
        float projection = dot(center, n);
        float slack = half_height * abs(dot(axis, n)) + dot(extent, abs(n)) - abs(projection);
        if (slack < depth)
        {
            depth = slack;
            normal = projection <= 0.0f ? n : -n;
            feature = candidate;
        }
    }
}

JphPairMetric jph_capsule_capsule_metric(float3 centerA, float4 rotationA, float halfA, float radiusA,
                                  float3 centerB, float4 rotationB, float halfB, float radiusB)
{
    float3 relative = centerB - centerA;
    float scale = max(jph_capsule_max_abs(relative), max(max(halfA, halfB), max(radiusA, radiusB)));
    float3 center = relative / scale;
    float3 axisA = jph_rotate(rotationA, float3(0.0f, 1.0f, 0.0f));
    float3 axisB = jph_rotate(rotationB, float3(0.0f, 1.0f, 0.0f));
    float ha = halfA / scale;
    float hb = halfB / scale;
    float ra = radiusA / scale;
    float rb = radiusB / scale;
    float3 p, q;
    jph_capsule_segment_closest(-ha * axisA, ha * axisA, center - hb * axisB, center + hb * axisB, p, q);
    float3 delta = q - p;
    float distance = jph_capsule_length(delta);
    float3 normal;
    float3 axis_cross = cross(axisA, axisB);
    bool parallel = jph_capsule_max_abs(axis_cross) == 0.0f;
    if (distance > 0.0f)
        normal = jph_capsule_unit(delta);
    else if (halfA > 0.0f && halfB > 0.0f && !parallel)
        normal = jph_capsule_unit(axis_cross);
    else
    {
        float3 radial = halfA > 0.0f ? jph_rotate(rotationA, float3(1.0f, 0.0f, 0.0f)) :
                        halfB > 0.0f ? jph_rotate(rotationB, float3(1.0f, 0.0f, 0.0f)) :
                        float3(1.0f, 0.0f, 0.0f);
        normal = jph_capsule_unit(radial);
    }
    JphPairMetric metric;
    metric.normal = normal;
    metric.separation = -(ra + rb - distance) * scale;
    metric.pointA = centerA + (p + ra * normal) * scale;
    metric.pointB = centerA + (q - rb * normal) * scale;
    metric.feature = 2048u;
    metric.status = 1u;
    return metric;
}

JphBoxManifold jph_capsule_capsule(float3 centerA, float4 rotationA, float halfA, float radiusA,
                                  float3 centerB, float4 rotationB, float halfB, float radiusB,
                                  float speculative_distance)
{
    JphPairMetric primary = jph_capsule_capsule_metric(centerA, rotationA, halfA, radiusA,
                                                                       centerB, rotationB, halfB, radiusB);
    float3 relative = centerB - centerA;
    float scale = max(jph_capsule_max_abs(relative), max(max(halfA, halfB), max(radiusA, radiusB)));
    float3 center = relative / scale;
    float3 axisA = jph_rotate(rotationA, float3(0.0f, 1.0f, 0.0f));
    float3 axisB = jph_rotate(rotationB, float3(0.0f, 1.0f, 0.0f));
    float ha = halfA / scale;
    float hb = halfB / scale;
    float ra = radiusA / scale;
    float rb = radiusB / scale;
    bool parallel = jph_capsule_max_abs(cross(axisA, axisB)) == 0.0f;
    float3 normal = primary.normal;
    JphBoxManifold result;
    result.normal = normal;
    result.penetration = -primary.separation;
    result.count = 0u;
    if (result.penetration < -speculative_distance)
        return result;
    result.count = 1u;
    result.points[0].pointA = primary.pointA;
    result.points[0].pointB = primary.pointB;
    result.points[0].depth = result.penetration;
    result.points[0].feature = primary.feature;
    if (halfA > 0.0f && halfB > 0.0f && parallel)
    {
        float aa = dot(axisA, axisA);
        float axial_center = dot(center, axisA) / aa;
        float axial_scale = dot(axisB, axisA) / aa;
        float low = max(-ha, axial_center - hb * abs(axial_scale));
        float high = min(ha, axial_center + hb * abs(axial_scale));
        if (low < high)
        {
            result.count = 2u;
            for (uint point = 0u; point < 2u; ++point)
            {
                float t = point == 0u ? low : high;
                float3 a = t * axisA;
                float3 b = center + ((t - axial_center) / axial_scale) * axisB;
                result.points[point].pointA = centerA + (a + ra * normal) * scale;
                result.points[point].pointB = centerA + (b - rb * normal) * scale;
                result.points[point].depth = dot(a - b, normal) * scale + radiusA + radiusB;
                result.points[point].feature = 2049u + point;
            }
        }
    }
    return result;
}

JphBoxManifold jph_capsule_sphere(float3 center, float4 rotation, float half_height, float radius,
                                 float3 sphere_center, float sphere_radius, float speculative_distance)
{
    return jph_capsule_capsule(center, rotation, half_height, radius,
                               sphere_center, float4(0.0f, 0.0f, 0.0f, 1.0f), 0.0f, sphere_radius,
                               speculative_distance);
}

JphPairMetric jph_capsule_box_local_metric(float3 center, float3 axis, float h, float r,
                                          float3 extent, float box_radius)
{
    float3 a = center - h * axis;
    float3 b = center + h * axis;
    float3 p, q, normal;
    float penetration;
    uint feature = 0u;
    jph_capsule_segment_box_closest(a, b, extent, p, q);
    float core_distance = jph_capsule_length(q - p);
    // Coincident closest witnesses also establish intersection. At an exact
    // corner touch the separately rounded slab parameters may straddle a ULP;
    // neither a true overlap nor coincident witnesses define a distance normal.
    if (jph_capsule_core_box_intersects(a, b, extent) || core_distance == 0.0f)
    {
        float core_depth;
        jph_capsule_core_box_mtd(center, axis, h, extent, normal, core_depth, feature);
        // Closest points in the intersecting core would coincide and lose the
        // penetration. Retreat to tangency, recover true support witnesses,
        // then restore only the capsule segment's displacement.
        jph_capsule_segment_box_closest(a - normal * core_depth, b - normal * core_depth, extent, p, q);
        p += normal * core_depth;
        penetration = core_depth + r + box_radius;
    }
    else
    {
        float3 delta = q - p;
        normal = jph_capsule_unit(delta);
        penetration = r + box_radius - core_distance;
        uint outside = 0u;
        for (uint coordinate = 0u; coordinate < 3u; ++coordinate)
            if (p[coordinate] < -extent[coordinate] || p[coordinate] > extent[coordinate])
                outside |= 1u << coordinate;
        feature = 8u + outside;
    }
    JphPairMetric metric;
    metric.normal = normal;
    metric.separation = -penetration;
    metric.pointA = p + r * normal;
    metric.pointB = q - box_radius * normal;
    metric.feature = 2080u + feature;
    metric.status = 1u;
    return metric;
}

JphPairMetric jph_capsule_box_metric(float3 capsule_center, float4 capsule_rotation, float half_height, float radius,
                                    float3 box_center, float4 box_rotation, float4 box_geometry)
{
    float3 local_center = jph_inverse_rotate(box_rotation, capsule_center - box_center);
    float3 axis = jph_inverse_rotate(box_rotation, jph_rotate(capsule_rotation, float3(0.0f, 1.0f, 0.0f)));
    float3 core = max(box_geometry.xyz - box_geometry.w, float3(0.0f));
    float scale = max(jph_capsule_max_abs(local_center),
                      max(jph_capsule_max_abs(core), max(half_height, max(radius, box_geometry.w))));
    JphPairMetric metric = jph_capsule_box_local_metric(local_center / scale, axis, half_height / scale,
                                                       radius / scale, core / scale, box_geometry.w / scale);
    metric.normal = jph_rotate(box_rotation, metric.normal);
    metric.separation *= scale;
    metric.pointA = box_center + jph_rotate(box_rotation, metric.pointA * scale);
    metric.pointB = box_center + jph_rotate(box_rotation, metric.pointB * scale);
    return metric;
}

JphBoxManifold jph_capsule_box(float3 capsule_center, float4 capsule_rotation, float half_height, float radius,
                              float3 box_center, float4 box_rotation, float4 box_geometry,
                              float speculative_distance)
{
    float3 local_center = jph_inverse_rotate(box_rotation, capsule_center - box_center);
    float3 axis = jph_inverse_rotate(box_rotation, jph_rotate(capsule_rotation, float3(0.0f, 1.0f, 0.0f)));
    float3 core = max(box_geometry.xyz - box_geometry.w, float3(0.0f));
    float scale = max(jph_capsule_max_abs(local_center),
                      max(jph_capsule_max_abs(core), max(half_height, max(radius, box_geometry.w))));
    float3 center = local_center / scale;
    float3 extent = core / scale;
    float h = half_height / scale;
    float r = radius / scale;
    float box_radius = box_geometry.w / scale;
    float spec = speculative_distance / scale;
    JphPairMetric primary = jph_capsule_box_local_metric(center, axis, h, r, extent, box_radius);
    float3 normal = primary.normal;
    float penetration = -primary.separation;
    // AABox::GetSupportingFace chooses the most aligned actual face. A
    // closest witness rounded just beyond an edge must not turn a near-face
    // contact into an edge-only manifold. Projection and finite core-face
    // clipping below still require every additional witness to lie on both
    // actual surfaces; this selection cannot create a primary hit.
    uint face_axis = 0u;
    if (abs(normal.y) > abs(normal.x))
        face_axis = 1u;
    if (abs(normal.z) > abs(normal[face_axis]))
        face_axis = 2u;
    JphBoxManifold result;
    result.normal = jph_rotate(box_rotation, normal);
    result.penetration = penetration * scale;
    result.count = 0u;
    if (result.penetration < -speculative_distance)
        return result;
    JphBoxPoint candidates[3];
    candidates[0].pointA = primary.pointA;
    candidates[0].pointB = primary.pointB;
    candidates[0].depth = dot(candidates[0].pointA - candidates[0].pointB, normal);
    candidates[0].feature = primary.feature;
    uint count = 1u;
    float axial_projection = dot(axis, normal);
    float3 radial = normal - axis * (axial_projection / dot(axis, axis));
    // CapsuleShape::GetSupportingFace uses .02m projected endpoint slop,
    // not a fixed angle. Project the radius into the true cylindrical plane.
    if (jph_capsule_max_abs(radial) > 0.0f &&
        2.0f * half_height * abs(axial_projection) < 0.02f)
    {
        float3 side = center + r * jph_capsule_unit(radial);
        float3 face_normal = float3(0.0f);
        face_normal[face_axis] = normal[face_axis] > 0.0f ? -1.0f : 1.0f;
        float3 plane = face_normal * (extent[face_axis] + box_radius);
        float denominator = dot(face_normal, normal);
        float t0 = dot(plane - side, face_normal) / denominator;
        float t1 = -dot(axis, face_normal) / denominator;
        float3 projected = side + t0 * normal;
        float3 projected_axis = axis + t1 * normal;
        float low = -h;
        float high = h;
        bool valid = true;
        for (uint coordinate = 0u; coordinate < 3u; ++coordinate)
            if (coordinate != face_axis)
                valid = valid &&
                    jph_capsule_clip_parameter(extent[coordinate] + projected[coordinate], projected_axis[coordinate], low, high) &&
                    jph_capsule_clip_parameter(extent[coordinate] - projected[coordinate], -projected_axis[coordinate], low, high);
        valid = valid && jph_capsule_clip_parameter(spec - t0, -t1, low, high);
        if (valid)
            for (uint point = 0u; point < 2u; ++point)
            {
                float s = point == 0u ? low : high;
                candidates[count].pointA = side + s * axis;
                candidates[count].pointB = projected + s * projected_axis;
                candidates[count].depth = -(t0 + s * t1);
                candidates[count].feature = 2112u + face_axis * 2u + point;
                ++count;
            }
    }
    // Keep the exact primary among at most three candidates. Equal-depth side
    // contacts retain the two extreme support points; a tilted cap's deeper
    // primary cannot be replaced by a shallower approximate supporting line.
    float3 patch_center = float3(0.0f);
    for (uint point = 0u; point < count; ++point)
        patch_center += 0.5f * (candidates[point].pointA + candidates[point].pointB);
    patch_center /= float(count);
    // Tolerances belong to the contact, not a distant box extent or the full
    // capsule length. A million-unit segment on a small platform still has a
    // small contact patch and must retain its distinct support endpoints.
    float depth_scale = max(max(r, box_radius), abs(penetration));
    float depth_tolerance = 8.0f * 1.192092896e-7f * depth_scale;
    float point_scale = max(r, box_radius);
    uint first = 0u;
    float first_distance = -1.0f;
    for (uint point = 0u; point < count; ++point)
    {
        float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - patch_center;
        point_scale = max(point_scale, jph_capsule_max_abs(delta));
        delta -= normal * dot(delta, normal);
        float distance = dot(delta, delta);
        if (candidates[point].depth > candidates[first].depth + depth_tolerance ||
            (abs(candidates[point].depth - candidates[first].depth) <= depth_tolerance && distance > first_distance))
        {
            first = point;
            first_distance = distance;
        }
    }
    uint second = first;
    float longest = 0.0f;
    float3 first_midpoint = 0.5f * (candidates[first].pointA + candidates[first].pointB);
    for (uint point = 0u; point < count; ++point)
    {
        float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - first_midpoint;
        delta -= normal * dot(delta, normal);
        float distance = dot(delta, delta);
        if (distance > longest)
        {
            longest = distance;
            second = point;
        }
    }
    float point_tolerance = 32.0f * 1.192092896e-7f * point_scale;
    result.count = longest > point_tolerance * point_tolerance ? 2u : 1u;
    for (uint point = 0u; point < result.count; ++point)
    {
        JphBoxPoint contact = candidates[point == 0u ? first : second];
        contact.pointA = box_center + jph_rotate(box_rotation, contact.pointA * scale);
        contact.pointB = box_center + jph_rotate(box_rotation, contact.pointB * scale);
        contact.depth *= scale;
        result.points[point] = contact;
    }
    return result;
}

JphPairMetric jph_finalize_pair_metric(JphPairMetric metric)
{
    // A nonrepresentable result is not a separated pair or a query miss.
    if (metric.status == 1u &&
        (!all(isfinite(metric.normal)) || !isfinite(metric.separation) ||
         !all(isfinite(metric.pointA)) || !all(isfinite(metric.pointB))))
        metric.status = 2u;
    return metric;
}

// Unified primary for every admitted primitive pair. No layer/spec/consumer
// policy is applied here. Query consumers inspect status before using fields.
JphPairMetric jph_pair_metric(uint typeA, float3 ca, float4 qa, float4 ga,
                               uint typeB, float3 cb, float4 qb, float4 gb)
{
    JphPairMetric metric = jph_empty_pair_metric();
    float penetration;
    if (typeA == ShapeSphere && typeB == ShapeSphere)
    {
        jph_sphere_sphere(ca, ga.w, cb, gb.w, metric.normal, penetration, metric.pointA, metric.pointB);
        metric.separation = -penetration;
        metric.status = 1u;
        return jph_finalize_pair_metric(metric);
    }
    if (typeA == ShapeSphere && typeB == ShapeBox)
    {
        jph_sphere_box(ca, ga.w, cb, qb, gb, metric.normal, penetration, metric.pointA, metric.pointB, metric.feature);
        metric.separation = -penetration;
        metric.status = 1u;
        return jph_finalize_pair_metric(metric);
    }
    if (typeA == ShapeBox && typeB == ShapeSphere)
    {
        jph_sphere_box(cb, gb.w, ca, qa, ga, metric.normal, penetration, metric.pointA, metric.pointB, metric.feature);
        metric.separation = -penetration;
        metric.status = 1u;
        return jph_finalize_pair_metric(jph_flip_pair_metric(metric));
    }
    if (typeA == ShapeBox && typeB == ShapeBox)
        return jph_finalize_pair_metric(jph_box_box_metric(ca, qa, ga, cb, qb, gb));
    if (typeA == ShapeCapsule && typeB == ShapeCapsule)
        return jph_finalize_pair_metric(jph_capsule_capsule_metric(ca, qa, ga.y, ga.w, cb, qb, gb.y, gb.w));
    if (typeA == ShapeCapsule && typeB == ShapeSphere)
        return jph_finalize_pair_metric(jph_capsule_capsule_metric(ca, qa, ga.y, ga.w, cb, qb, 0.0f, gb.w));
    if (typeA == ShapeSphere && typeB == ShapeCapsule)
        return jph_finalize_pair_metric(jph_flip_pair_metric(jph_capsule_capsule_metric(cb, qb, gb.y, gb.w, ca, qa, 0.0f, ga.w)));
    if (typeA == ShapeCapsule && typeB == ShapeBox)
        return jph_finalize_pair_metric(jph_capsule_box_metric(ca, qa, ga.y, ga.w, cb, qb, gb));
    if (typeA == ShapeBox && typeB == ShapeCapsule)
        return jph_finalize_pair_metric(jph_flip_pair_metric(jph_capsule_box_metric(cb, qb, gb.y, gb.w, ca, qa, ga)));
    return jph_finalize_pair_metric(metric);
}

float jph_combine_material(float value_a, uint mode_a, float value_b, uint mode_b)
{
    // The immutable model boundary validates the four supported modes.
    uint mode = max(mode_a, mode_b);
    if (mode == MaterialCombineAverage)
        return 0.5f * (value_a + value_b);
    if (mode == MaterialCombineMinimum)
        return min(value_a, value_b);
    if (mode == MaterialCombineMultiply)
        return value_a * value_b;
    return max(value_a, value_b);
}

void jph_solve_friction(BodyModel ma, inout BodyState sa, float3 ra,
                        BodyModel mb, inout BodyState sb, float3 rb,
                        float3 normal, float maximum_impulse, inout float2 impulse)
{
    float3 tangent1, tangent2;
    jph_tangents(normal, tangent1, tangent2);

    // ContactConstraintManager solves linear friction at the average contact
    // midpoint, with a circular bound from the previous iteration's SUM of
    // normal impulses. The manifold owns this pair of accumulated lambdas.
    float lambda1 = impulse.x
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent1) * jph_jv(ma, sa, ra, mb, sb, rb, tangent1);
    float lambda2 = impulse.y
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent2) * jph_jv(ma, sa, ra, mb, sb, rb, tangent2);
    float friction_sq = lambda1 * lambda1 + lambda2 * lambda2;
    if (friction_sq > maximum_impulse * maximum_impulse)
    {
        float scale = maximum_impulse / sqrt(friction_sq);
        lambda1 *= scale;
        lambda2 *= scale;
    }
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent1, lambda1 - impulse.x);
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent2, lambda2 - impulse.y);
    impulse = float2(lambda1, lambda2);
}

void jph_solve_angular_friction(BodyModel ma, inout BodyState sa, BodyModel mb, inout BodyState sb,
                                float3 normal, float maximum_impulse, inout float impulse)
{
    float3 inverse_axis_a = jph_inverse_inertia(ma, sa, normal);
    float3 inverse_axis_b = jph_inverse_inertia(mb, sb, normal);
    float inverse_effective_mass = dot(normal, inverse_axis_a + inverse_axis_b);
    float effective_mass = inverse_effective_mass > 0.0f ? 1.0f / inverse_effective_mass : 0.0f;
    float3 angular_a = jph_nonstatic(ma) ? sa.angularVelocity.xyz : float3(0.0f);
    float3 angular_b = jph_nonstatic(mb) ? sb.angularVelocity.xyz : float3(0.0f);
    float lambda = clamp(impulse + effective_mass * dot(normal, angular_a - angular_b),
                         -maximum_impulse, maximum_impulse);
    float delta = lambda - impulse;
    if (jph_dynamic(ma))
        sa.angularVelocity.xyz -= delta * inverse_axis_a;
    if (jph_dynamic(mb))
        sb.angularVelocity.xyz += delta * inverse_axis_b;
    impulse = lambda;
}

float4 jph_solve_normal_block(BodyModel ma, BodyState sa, BodyModel mb, BodyState sb,
                               float3 normal, float3 ja[4], float3 jb[4],
                               float4 old_impulse, float4 bias, uint count)
{
    // All points share n; their midpoint offsets vary in only two tangent
    // coordinates. Thus J, and this pair-local PSD K, have rank at most three.
    // The unilateral QP has a minimizer on an independent active basis of
    // size <= 3, even if a four-point force distribution is non-unique.
    float4 k[4];
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        k[point] = float4(0.0f);
    float inverse_mass = jph_inverse_mass_axis(ma, normal) + jph_inverse_mass_axis(mb, normal);
    for (uint column = 0u; column < MaxContactPointsPerPair; ++column)
    {
        if (column >= count)
            break;
        float3 inverse_a = jph_inverse_inertia(ma, sa, ja[column]);
        float3 inverse_b = jph_inverse_inertia(mb, sb, jb[column]);
        for (uint row = 0u; row < MaxContactPointsPerPair; ++row)
        {
            if (row > column)
                break;
            float value = inverse_mass + dot(ja[row], inverse_a) + dot(jb[row], inverse_b);
            k[row][column] = value;
            k[column][row] = value;
        }
    }

    float3 linear_a = jph_nonstatic(ma) ? sa.linearVelocity.xyz : float3(0.0f);
    float3 linear_b = jph_nonstatic(mb) ? sb.linearVelocity.xyz : float3(0.0f);
    float3 angular_a = jph_nonstatic(ma) ? sa.angularVelocity.xyz : float3(0.0f);
    float3 angular_b = jph_nonstatic(mb) ? sb.angularVelocity.xyz : float3(0.0f);
    float linear_jv = dot(normal, linear_a - linear_b);
    float4 q = float4(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < count)
            q[point] = linear_jv + dot(ja[point], angular_a) - dot(jb[point], angular_b)
                - bias[point] + dot(k[point], old_impulse);

    // Equilibrate only AFTER reconstructing q with physical K and old lambda.
    // Dii=sqrt(Kii), C=D^-1 K D^-1, r=D^-1 q and y=D lambda.
    // Reuse k for C. Its unit diagonal makes rank tests independent of the
    // different mass/inertia scales of otherwise independent contact rows.
    float4 inverse_scale = float4(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < count)
            inverse_scale[point] = k[point][point] > 0.0f ? 1.0f / sqrt(k[point][point]) : 0.0f;
    for (uint column = 0u; column < MaxContactPointsPerPair; ++column)
    {
        if (column >= count)
            break;
        for (uint row = 0u; row < MaxContactPointsPerPair; ++row)
        {
            if (row > column)
                break;
            float value = k[row][column] * inverse_scale[row] * inverse_scale[column];
            k[row][column] = value;
            k[column][row] = value;
        }
        k[column][column] = inverse_scale[column] > 0.0f ? 1.0f : 0.0f;
    }
    float4 scaled_q = q * inverse_scale;

    // Empty set plus 4 singleton, 6 pair and 4 triple bases: at most fifteen
    // candidates. Zero is a normal feasible QP candidate, not a PGS fallback.
    float4 best = float4(0.0f);
    for (uint mask = 1u; mask < 16u; ++mask)
    {
        if (mask >= (1u << count))
            break;
        uint size = (mask & 1u) + ((mask >> 1u) & 1u) + ((mask >> 2u) & 1u) + ((mask >> 3u) & 1u);
        if (size > 3u)
            continue;
        uint indices[3];
        uint next = 0u;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            if ((mask & (1u << point)) != 0u)
                indices[next++] = point;

        uint i = indices[0];
        // Relative to C's unit diagonal, LDL pivots identify duplicate or
        // collinear rows without treating a differently scaled row as null.
        float rank_threshold = 1e-6f;
        float d0 = k[i][i];
        if (d0 <= rank_threshold)
            continue;
        float y0 = scaled_q[i];
        float z0 = y0 / d0;
        float3 solution = float3(z0, 0.0f, 0.0f);
        if (size > 1u)
        {
            uint j = indices[1];
            float l10 = k[j][i] / d0;
            float d1 = k[j][j] - l10 * k[j][i];
            if (d1 <= rank_threshold)
                continue;
            float y1 = scaled_q[j] - l10 * y0;
            float z1 = y1 / d1;
            solution.y = z1;
            solution.x = z0 - l10 * solution.y;
            if (size > 2u)
            {
                uint third = indices[2];
                float l20 = k[third][i] / d0;
                float l21 = (k[third][j] - l20 * k[j][i]) / d1;
                float d2 = k[third][third] - l20 * k[third][i] - l21 * l21 * d1;
                if (d2 <= rank_threshold)
                    continue;
                float y2 = scaled_q[third] - l20 * y0 - l21 * y1;
                solution.z = y2 / d2;
                solution.y = z1 - l21 * solution.z;
                solution.x = z0 - l10 * solution.y - l20 * solution.z;
            }
        }
        if (any(solution < float3(0.0f)))
            continue;
        float4 candidate = float4(0.0f);
        candidate[indices[0]] = solution.x;
        if (size > 1u)
            candidate[indices[1]] = solution.y;
        if (size > 2u)
            candidate[indices[2]] = solution.z;
        // E(y)=.5*y^T*C*y-r^T*y equals the physical impulse objective.
        // Compare E(candidate)-E(best) directly: subtracting two large total
        // energies can erase a real improvement on a smaller-scale row.
        float4 sum = candidate + best;
        float4 response = float4(dot(k[0], sum), dot(k[1], sum), dot(k[2], sum), dot(k[3], sum));
        float energy_delta = dot(candidate - best, 0.5f * response - scaled_q);
        if (energy_delta < 0.0f)
            best = candidate;
    }
    return best * inverse_scale;
}

void jph_solve_position_points(BodyModel ma, inout BodyState sa, BodyModel mb, inout BodyState sb,
                               float3 normal, float3 pa, float3 pb,
                               float penetration_slop, float maximum_correction, float baumgarte)
{
    float3 midpoint = 0.5f * (pa + pb);
    float3 ra = midpoint - sa.positionCOM.xyz;
    float3 rb = midpoint - sb.positionCOM.xyz;
    float separation = max(dot(pb - pa, normal) + penetration_slop, -maximum_correction);
    if (separation >= 0.0f)
        return;
    float lambda = -jph_effective_mass(ma, sa, ra, mb, sb, rb, normal) * baumgarte * separation;
    if (jph_dynamic(ma))
    {
        float3 angular_step = -lambda * jph_inverse_inertia(ma, sa, cross(ra, normal));
        sa.positionCOM.xyz -= lambda * jph_inverse_linear_mass(ma, normal);
        sa.rotation = jph_rotation_step(sa.rotation, angular_step);
        jph_project_motion(ma, sa);
    }
    if (jph_dynamic(mb))
    {
        float3 angular_step = lambda * jph_inverse_inertia(mb, sb, cross(rb, normal));
        sb.positionCOM.xyz += lambda * jph_inverse_linear_mass(mb, normal);
        sb.rotation = jph_rotation_step(sb.rotation, angular_step);
        jph_project_motion(mb, sb);
    }
    // Split position correction deliberately leaves the physical velocities alone.
}

#endif // JPH_GPU_RIGID_MATH_SLANG
