// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// Slang adaptation of MuJoCo Warp _src/constraint.py connect and weld rows.

#include "model.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"
#include "prescribed_motion.slang"

static const uint SpatialEqualityWords = 9;
static const uint SceneJointWords = 10;
static const uint SceneHinge = 1;
static const uint SceneSlider = 2;
static const uint SceneDistance = 3;
static const uint SceneSwingTwist = 4;

struct HingeDrive
{
    float4 motor;
    float4 passive;
};

[[vk::binding(1, 0)]] StructuredBuffer<uint4> spatialModel;
[[vk::binding(2, 0)]] StructuredBuffer<uint> equalityActive;
[[vk::binding(3, 0)]] StructuredBuffer<Pose> bodyPose;
[[vk::binding(4, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(6, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(9, 0)]] StructuredBuffer<SpatialMotion> centeredDofDerivative;
[[vk::binding(10, 0)]] StructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(11, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(12, 0)]] StructuredBuffer<BodyModel> sceneBodies;
[[vk::binding(13, 0)]] StructuredBuffer<PrescribedMotion> scenePrescribed;
[[vk::binding(14, 0)]] StructuredBuffer<uint> sceneFixedSteps;
[[vk::binding(15, 0)]] RWStructuredBuffer<SpatialMotion> sceneBodyVelocity;
[[vk::binding(16, 0)]] StructuredBuffer<ModelDimensions> sceneDimensions;
[[vk::binding(17, 0)]] StructuredBuffer<uint4> sceneParameters;

struct SpatialEndpoint
{
    float3 point;
    float3 offset;
    float3 pointVelocity;
    float3 omega;
    float4 orientation;
};

float4 conjugateQuaternion(float4 q)
{
    return float4(q.x, -q.yzw);
}

SpatialEndpoint equalityEndpoint(uint base, uint body, uint welded, uint root, float3 localPoint, float4 localFrame)
{
    const Pose pose = bodyPose[base + body];
    const SpatialMotion weldedMotion = bodyVelocity[base + welded];
    const float3 point = pose.position.xyz + rotateVector(localPoint, pose.rotation);
    const float3 offset = point - subtreeCom[base + root].xyz;
    return {point, offset, weldedMotion.linear.xyz + cross(weldedMotion.angular.xyz, offset),
            bodyVelocity[base + body].angular.xyz, mulQuat(pose.rotation, localFrame)};
}

float3 pointJacobian(SpatialMotion motion, SpatialEndpoint endpoint)
{
    return motion.linear.xyz + cross(motion.angular.xyz, endpoint.offset);
}

float3 pointJacobianDerivative(SpatialMotion motion, SpatialMotion derivative, SpatialEndpoint endpoint)
{
    return pointJacobian(derivative, endpoint) + cross(motion.angular.xyz, endpoint.pointVelocity);
}

struct AxialRow
{
    float4 response; // Distance, reference acceleration, inverse regularization, force cap.
    float direction;
    uint kind;
};

struct AxialRows
{
    AxialRow rows[5];
    uint count;
};

float wrapHingeAngle(float angle)
{
    const float turn = 2.0 * PhysicsPi;
    return angle - turn * floor((angle + PhysicsPi) / turn);
}

void appendAxisResponse(inout AxialRows result, float reference, float diagonal, uint kind, float cap)
{
    result.rows[result.count++] = {float4(0.0, reference, diagonal, cap), 1.0, kind};
}

void appendAxisLimit<let Linear : bool>(inout AxialRows result, float2 limit, float value, float speed, float weight,
                                       float timestep, uint flags, ConstraintParameters parameters, float2 spring,
                                       bool lowerCanBind)
{
    const float predicted = value + timestep * speed;
    const bool locked = limit.x == limit.y;
    const bool upper = !locked && (value > limit.y || (value >= limit.x && predicted > limit.y));
    const bool lower = lowerCanBind && (value < limit.x || predicted < limit.x);
    if (locked || upper || lower)
    {
        const float direction = upper ? -1.0 : 1.0;
        const float error = direction * (value - (upper ? limit.y : limit.x));
        const float velocity = direction * speed;
        const uint kind = locked ? EqualityRow : LimitRow;
        float reference, diagonal;
        if (spring.x > 0.0)
        {
            const float damping = spring.y + timestep * spring.x;
            diagonal = timestep * damping;
            reference = -(spring.x * error + damping * velocity) / diagonal;
        }
        else if (Linear && !locked)
        {
            // Limit the end-of-substep velocity to the remaining travel. Correct any
            // existing penetration gradually instead of injecting a restitution impulse.
            const float gap = error < 0.0 ? 0.2 * error : error;
            diagonal = constraintDiagonal(weight, MaxImpedance);
            reference = -(gap + timestep * velocity) / (timestep * timestep);
        }
        else
        {
            const float3 kbi = impedanceParameters(parameters, error, timestep, false);
            const ConstraintRow row = makeConstraintRow(uint4(0u), kind, kbi, error, 0.0, velocity, weight,
                                                       0.0, 0.0, timestep, flags, parameters.reference.x);
            reference = row.state.w;
            diagonal = row.diagonal;
            if (!locked && velocity < 0.0)
                reference = max(reference, -(error + timestep * velocity) / (timestep * timestep));
        }
        result.rows[result.count++] = {float4(error, reference, diagonal, 0.0), direction, kind};
    }
}

AxialRows hingeRows(uint4 controls, HingeDrive drive, float angle, float speed, float weight, float timestep, uint flags,
                    ConstraintParameters parameters)
{
    AxialRows result;
    result.count = 0;
    const float2 limit = asfloat(controls.yz);
    const float4 motor = drive.motor;
    const float3 passive = drive.passive.xyz;
    const bool limited = controls.x != 0;
    if (limited)
        appendAxisLimit<false>(result, limit, angle, speed, weight, timestep, flags, parameters, float2(0.0), true);
    const float motorDiagonal = timestep * (motor.z + motor.y * timestep);
    if (motorDiagonal > 0.0 && motor.w > 0.0)
    {
        // Implicit spring and damper use the end-of-substep pose and speed.
        const float target = limited ? clamp(motor.x, limit.x, limit.y) : motor.x;
        const float error = wrapHingeAngle(target - angle);
        const float torque = motor.y * (error - timestep * speed) - motor.z * speed;
        appendAxisResponse(result, torque / motorDiagonal, motorDiagonal, FrictionRow, motor.w);
    }
    if (passive.x > 0.0)
        appendAxisResponse(result, -speed / timestep, passive.x * timestep, EqualityRow, 0.0);
    if (passive.y > 0.0)
        appendAxisResponse(result, -speed / timestep, 1.0 / (weight * timestep), FrictionRow, passive.y);
    if (passive.z > 0.0)
        appendAxisResponse(result, 0.0, passive.z, EqualityRow, 0.0);
    return result;
}

struct AngularRow
{
    float3 axis; // Jacobian direction in the connected endpoint frame.
    float4 response;
    uint kind;
};

struct AngularRows
{
    AngularRow rows[6];
    uint count;
};

struct SwingTwistCoordinates
{
    float3 swing; // W, Y, Z. The twist component rotates around X.
    float twist;
};

SwingTwistCoordinates swingTwistCoordinates(float4 q)
{
    const float h = length(q.xy);
    if (h < 1e-7)
        return {float3(0.0, q.zw), 0.0};
    const float2 swingPlane = (q.x * q.zw + q.y * float2(-q.w, q.z)) / h;
    return {float3(h, swingPlane), wrapHingeAngle(2.0 * atan2(q.y, q.x))};
}

float wrapTwistToRange(float value, float2 bounds)
{
    const float center = 0.5 * (bounds.x + bounds.y);
    return center + wrapHingeAngle(value - center);
}

float4 limitSwingTwistTarget(float4 q, float4 limits, bool elliptic)
{
    const SwingTwistCoordinates coordinates = swingTwistCoordinates(q);
    float3 swing = coordinates.swing;
    if (elliptic)
    {
        const float ellipse = length(swing.yz / sin(0.5 * limits.wz));
        if (ellipse > 1.0)
        {
            swing.yz /= ellipse;
            swing.x = sqrt(max(0.0, 1.0 - dot(swing.yz, swing.yz)));
        }
    }
    else
    {
        const float2 denominator = swing.x == 0.0 ? float2(swing.y == 0.0, swing.z == 0.0) : float2(swing.x);
        const float2 angle = atan2(swing.yz, denominator);
        const float2 bounded = clamp(angle, -0.5 * limits.wz, 0.5 * limits.wz);
        if (any(angle != bounded))
        {
            const float2 cs = cos(bounded);
            const float2 sn = sin(bounded);
            swing = normalize(float3(cs.x * cs.y, sn.x * cs.y, cs.x * sn.y));
        }
    }
    const float angle = 0.5 * clamp(wrapTwistToRange(coordinates.twist, limits.xy), limits.x, limits.y);
    return mulQuat(float4(swing.x, 0.0, swing.yz), float4(cos(angle), sin(angle), 0.0, 0.0));
}

void appendAngularLimit(inout AngularRows result, float value, float2 bounds, float3 gradient,
                        float3 relativeVelocity, float weight, float timestep, bool hasLower = true,
                        float2 spring = float2(0.0))
{
    const float width = length(gradient);
    if (width < 1e-7)
        return;
    const float3 axis = gradient / width;
    const float speed = dot(axis, relativeVelocity);
    const float predicted = value + timestep * speed * width;
    const bool locked = bounds.x == bounds.y;
    const bool upper = !locked && (value > bounds.y || (value >= bounds.x && predicted > bounds.y));
    const bool lower = hasLower && (value < bounds.x || predicted < bounds.x);
    if (!locked && !upper && !lower)
        return;
    const float direction = upper ? -1.0 : 1.0;
    const float distance = direction * (value - (upper ? bounds.y : bounds.x)) / width;
    const float gap = locked || distance < 0.0 ? 0.2 * distance : distance;
    const float damping = spring.y + timestep * spring.x;
    const float diagonal = spring.x > 0.0 ? timestep * damping : constraintDiagonal(weight, MaxImpedance);
    const float reference = spring.x > 0.0 ? -(spring.x * distance + damping * direction * speed) / diagonal :
        -(gap + timestep * direction * speed) / (timestep * timestep);
    result.rows[result.count++] = {direction * axis, float4(distance, reference, diagonal, 0.0),
                                   locked ? EqualityRow : LimitRow};
}

struct SwingTwistDifferential
{
    float3 swing;
    float3 h, y, z, twist;
};

SwingTwistDifferential swingTwistDifferential(float4 rotation, float3 velocity, float timestep)
{
    // Evaluate the boundary normal midway along the free angular displacement.
    const float speed = length(velocity);
    const float4 midpoint = speed > 0.0 ?
        mulQuat(axisAngleQuat(velocity / speed, 0.5 * timestep * speed), rotation) : rotation;
    const float3 swing = swingTwistCoordinates(midpoint).swing;
    const float h = swing.x;
    const float3 dw = -0.5 * midpoint.yzw;
    const float3 dx = 0.5 * float3(midpoint.x, midpoint.w, -midpoint.z);
    const float3 dy = 0.5 * float3(-midpoint.w, midpoint.x, midpoint.y);
    const float3 dz = 0.5 * float3(midpoint.z, -midpoint.y, midpoint.x);
    const float inverseH = 1.0 / max(h, 1e-7);
    const float3 dh = (midpoint.x * dw + midpoint.y * dx) * inverseH;
    const float3 y = (midpoint.x * dy + midpoint.z * dw - midpoint.y * dz - midpoint.w * dx - swing.y * dh) * inverseH;
    const float3 z = (midpoint.x * dz + midpoint.w * dw + midpoint.y * dy + midpoint.z * dx - swing.z * dh) * inverseH;
    const float3 twist = h > 1e-7 ? 2.0 * (midpoint.x * dx - midpoint.y * dw) * inverseH * inverseH : float3(1, 0, 0);
    return {swing, dh, y, z, twist};
}

// Match the authored angular frame and cone/pyramid conventions with a device-side
// Jacobian of the swing-twist coordinates. Limits and motors share the sparse solver.
AngularRows swingTwistRows(uint record, float4 relative, float3 relativeVelocity, float weight, float timestep)
{
    AngularRows result;
    result.count = 0u;
    const float4 angles = asfloat(sceneParameters[record]); // Twist min/max, plane (Z), normal (Y).
    const uint3 modes = sceneParameters[record + 1].xyz;
    const SwingTwistCoordinates coordinates = swingTwistCoordinates(relative);
    const float3 swing = coordinates.swing;
    const float2 swingLimits = angles.wz;
    // Evaluate the boundary normal halfway through the free angular displacement.
    // A normal frozen at the start of the step does spurious work while sliding
    // around a curved cone and systematically gains energy.
    const SwingTwistDifferential differential = swingTwistDifferential(relative, relativeVelocity, timestep);
    const float3 midSwing = differential.swing;
    const float h = midSwing.x;
    const float3 dh = differential.h;
    const float3 swingY = differential.y;
    const float3 swingZ = differential.z;
    const float3 twist = differential.twist;
    if (angles.x > -PhysicsPi || angles.y < PhysicsPi)
    {
        const float angle = wrapTwistToRange(coordinates.twist, angles.xy);
        appendAngularLimit(result, angle, angles.xy, twist, relativeVelocity, weight, timestep);
    }

    const bool elliptic = modes.x == 0 && all(swingLimits > 0.0) && any(swingLimits < PhysicsPi);
    if (elliptic)
    {
        const float2 radius = sin(0.5 * swingLimits);
        const float2 ellipse = swing.yz / radius;
        if (h > 1e-4)
        {
            const float2 gradient = 2.0 * midSwing.yz / (radius * radius);
            appendAngularLimit(result, dot(ellipse, ellipse), float2(0.0, 1.0),
                               gradient.x * swingY + gradient.y * swingZ, relativeVelocity, weight, timestep, false);
        }
        else
        {
            // At a half turn, the ellipse's squared coordinate has zero radial slope.
            const float2 direction = normalize(midSwing.yz);
            const float bound = 2.0 * asin(min(1.0, 1.0 / length(direction / radius)));
            appendAngularLimit(result, 2.0 * atan2(length(swing.yz), swing.x), float2(0.0, bound),
                               float3(0.0, direction), relativeVelocity, weight, timestep, false);
        }
    }
    else
    {
        const float3 derivatives[2] = {swingY, swingZ};
        for (uint axis = 0; axis < 2; ++axis)
        {
            const float part = midSwing[axis + 1];
            const float norm = h * h + part * part;
            if (swingLimits[axis] < PhysicsPi && norm > 1e-14)
            {
                const float currentPart = swing[axis + 1];
                const float angle = swing.x == 0.0 && currentPart == 0.0 ? 0.0 : 2.0 * atan2(currentPart, swing.x);
                const float3 gradient = (2.0 / norm) * (h * derivatives[axis] - part * dh);
                appendAngularLimit(result, angle, float2(-swingLimits[axis], swingLimits[axis]), gradient,
                                   relativeVelocity, weight, timestep);
            }
        }
    }

    const float4 springs = asfloat(sceneParameters[record + 2]);
    const float3 caps = asfloat(sceneParameters[record + 3].xyz);
    float4 offset = float4(0.0);
    if (modes.y >= 2 || modes.z >= 2)
    {
        const float4 target = limitSwingTwistTarget(asfloat(sceneParameters[record + 4]), angles, elliptic);
        offset = mulQuat(conjugateQuaternion(relative), dot(relative, target) < 0.0 ? -target : target);
    }
    const float3 velocity = asfloat(sceneParameters[record + 5].xyz);
    for (uint axis = 0; axis < 3; ++axis)
    {
        const bool isTwist = axis == 0;
        const uint mode = isTwist ? modes.z : modes.y;
        const float cap = mode == 0 ? caps.z : (isTwist ? caps.y : caps.x);
        const float2 spring = isTwist ? springs.zw : springs.xy;
        if (cap <= 0.0 || (mode == 2 && spring.x == 0.0) || (mode == 3 && all(spring == 0.0)))
            continue;
        float3 basis = float3(0.0);
        basis[axis] = 1.0;
        const float3 direction = rotateVector(basis, relative);
        const float speed = dot(direction, relativeVelocity);
        const float errorVelocity = ((mode & 1u) != 0 ? velocity[axis] : 0.0) - speed;
        float diagonal = 1.0 / (timestep * weight);
        float reference = errorVelocity / timestep;
        if (mode >= 2)
        {
            diagonal = timestep * (spring.y + timestep * spring.x);
            reference = (spring.x * (2.0 * offset[axis + 1] + timestep * errorVelocity) +
                spring.y * errorVelocity) / diagonal;
        }
        result.rows[result.count++] = {direction, float4(0.0, reference, diagonal, cap), FrictionRow};
    }
    return result;
}

void emitSpatialRows<let SceneJoint : bool>(uint3 id)
{
    const uint4 model = spatialModel[0]; // Descriptor count, then parameter or equality activation stride.
    if (id.x >= model.x)
        return;
    const uint record = 1 + (SceneJoint ? SceneJointWords : SpatialEqualityWords) * id.x;
    const uint4 source = spatialModel[record]; // equality id, first body, second body, Jacobian width
    if (!SceneJoint && equalityActive[id.y * model.y + source.x] == 0)
        return;
    const uint4 span = spatialModel[record + 1]; // first welded body, second welded body, support offset, height
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint bodyBase = id.y * config.x;
    const uint dofBase = id.y * config.y;
    const uint2 joint = SceneJoint ? spatialModel[record + 9].xy : uint2(0u);
    const uint jointType = joint.x;
    const bool hinge = jointType == SceneHinge;
    const bool slider = jointType == SceneSlider;
    const bool distance = jointType == SceneDistance;
    const bool swingTwist = jointType == SceneSwingTwist;
    const uint linearRows = distance ? 0u : (slider ? 2u : 3u);
    const uint angularRows = span.w - linearRows;
    const uint4 reference = spatialModel[record + 8]; // roots, inverse weights as float bits
    const float4 anchorA = asfloat(spatialModel[record + 4]);
    const float3 anchorB = asfloat(spatialModel[record + 5].xyz);
    const SpatialEndpoint a = equalityEndpoint(bodyBase, source.y, span.x, reference.x, anchorA.xyz,
                                               asfloat(spatialModel[record + 6]));
    const SpatialEndpoint b = equalityEndpoint(bodyBase, source.z, span.y, reference.y, anchorB,
                                               asfloat(spatialModel[record + 7]));
    const float4 orientationAInverse = conjugateQuaternion(a.orientation);
    const float4 orientationBInverse = conjugateQuaternion(b.orientation);
    const float4 relative = mulQuat(orientationBInverse, a.orientation);
    const float3 relativeVelocity = rotateVector(a.omega - b.omega, orientationBInverse);
    const float3 hingeAxis = rotateVector(float3(0.0, 0.0, 1.0), b.orientation);
    const float3 separation = a.point - b.point;
    const float3 pointVelocity = a.pointVelocity - b.pointVelocity;
    const float separationLength = distance ? length(separation) : 0.0;
    const float3 radial = separationLength > 0.0 ? separation / separationLength : float3(0.0, 1.0, 0.0);
    const float3 slide = slider ? rotateVector(-separation, orientationAInverse) : float3(0.0);
    const float3 slideVelocity = slider ?
        rotateVector(-pointVelocity + cross(a.omega, separation), orientationAInverse) : float3(0.0);
    const float speed = hinge ? dot(hingeAxis, a.omega - b.omega) :
                       (slider ? slideVelocity.z : dot(radial, pointVelocity));
    const ConstraintParameters parameters = {asfloat(spatialModel[record + 2]), asfloat(spatialModel[record + 3])};
    const float timestep = asfloat(settings[2 + id.y].x);
    const uint parameterIndex = id.y * model.y + joint.y;
    AxialRows axial;
    axial.count = 0;
    AngularRows swing;
    swing.count = 0;
    if (swingTwist)
        swing = swingTwistRows(parameterIndex, relative, relativeVelocity, asfloat(reference.w), timestep);
    else if (hinge)
    {
        const uint4 controls = sceneParameters[parameterIndex];
        const float theta = wrapHingeAngle(2.0 * atan2(relative.w, relative.x));
        const HingeDrive drive = {asfloat(sceneParameters[parameterIndex + 1]),
                                  asfloat(sceneParameters[parameterIndex + 2])};
        axial = hingeRows(controls, drive, theta, speed, asfloat(reference.w), timestep, config.w, parameters);
    }
    else if (slider || distance)
    {
        const uint4 controls = sceneParameters[parameterIndex];
        if (controls.x != 0)
        {
            const float2 limit = asfloat(controls.yz);
            const float2 spring = distance ? asfloat(sceneParameters[parameterIndex + 1].xy) : float2(0.0);
            // A range starting at zero permits its attachment points to cross.
            appendAxisLimit<true>(axial, limit, slider ? slide.z : separationLength, speed, asfloat(reference.z),
                                  timestep, config.w, parameters, spring, !distance || limit.x > 0.0);
        }
    }
    const uint totalRows = span.w + axial.count + swing.count;
    if (totalRows == 0)
        return;
    const uint2 block = reserveConstraintBlock(id.y, totalRows, source.w);
    if (block.x == NoIndex)
        return;
    const uint columnBase = id.y * limits.w + block.y;
    const float torqueScale = anchorA.w;
    float3 linearVelocity = float3(0.0);
    float3 angularVelocity = float3(0.0);
    float3 linearShift = float3(0.0);
    float3 angularShift = float3(0.0);
    float3 sliderAngularShift = float3(0.0);
    for (uint j = 0; j < source.w; ++j)
    {
        // A DOF shared by the endpoints contributes to both point Jacobians.
        const uint4 support = spatialModel[span.z + j]; // DOF, endpoint mask, quaternion owner or NoIndex
        const bool inA = (support.y & 1u) != 0;
        const bool inB = (support.y & 2u) != 0;
        const uint index = dofBase + support.x;
        const SpatialMotion motion = centeredDof[index];
        const SpatialMotion derivative = support.z == NoIndex ? centeredDofDerivative[index] :
            motionCross(bodyVelocity[bodyBase + support.z], motion);
        const float speed = qvel[index];
        const float3 linear = (inA ? pointJacobian(motion, a) : float3(0.0)) -
                              (inB ? pointJacobian(motion, b) : float3(0.0));
        const float3 linearDot = (inA ? pointJacobianDerivative(motion, derivative, a) : float3(0.0)) -
                                 (inB ? pointJacobianDerivative(motion, derivative, b) : float3(0.0));
        const float3 linearJacobian = slider ? rotateVector(
            -linear + (inA ? cross(motion.angular.xyz, separation) : float3(0.0)), orientationAInverse) : linear;
        for (uint axis = 0; axis < linearRows; ++axis)
            columns[columnBase + axis * source.w + j] = {support.x, linearJacobian[axis]};
        linearVelocity += linear * speed;
        linearShift += linearDot * speed;

        const float direction = float(inA) - float(inB);
        if (angularRows != 0u || swing.count != 0u)
            angularShift += direction * derivative.angular.xyz * speed;
        if (angularRows != 0u)
        {
            const float4 relativeAxis = mulQuat(mulQuat(orientationBInverse,
                float4(0.0, direction * motion.angular.xyz)), a.orientation);
            const float3 angular = (0.5 * torqueScale) * relativeAxis.yzw;
            for (uint axis = 0; axis < angularRows; ++axis)
                columns[columnBase + (axis + linearRows) * source.w + j] = {support.x, angular[axis]};
            angularVelocity += angular * speed;
        }
        if (slider && inA)
            sliderAngularShift += derivative.angular.xyz * speed;
        if (axial.count != 0)
        {
            const float value = hinge ? direction * dot(hingeAxis, motion.angular.xyz) :
                                (slider ? linearJacobian.z : dot(radial, linear));
            for (uint extra = 0; extra < axial.count; ++extra)
                columns[columnBase + (span.w + extra) * source.w + j] = {support.x, axial.rows[extra].direction * value};
        }
        if (swing.count != 0)
        {
            const float3 value = rotateVector(direction * motion.angular.xyz, orientationBInverse);
            for (uint extra = 0; extra < swing.count; ++extra)
                columns[columnBase + (span.w + extra) * source.w + j] = {support.x, dot(swing.rows[extra].axis, value)};
        }
    }

    float3 linearError = separation;
    float3 angularError = float3(0.0);
    float axialShift = dot(hingeAxis, angularShift) + dot(cross(b.omega, hingeAxis), a.omega - b.omega);
    if (angularRows != 0u)
    {
        angularError = torqueScale * relative.yzw;
        if (hinge)
            angularError.z = 0.0;
        const float4 delta = float4(0.0, a.omega - b.omega);
        const float4 aDot = 0.5 * mulQuat(float4(0.0, a.omega), a.orientation);
        const float4 bDotInv = conjugateQuaternion(0.5 * mulQuat(float4(0.0, b.omega), b.orientation));
        const float4 shift = mulQuat(mulQuat(bDotInv, delta), a.orientation) +
            mulQuat(mulQuat(orientationBInverse, float4(0.0, angularShift)), a.orientation) +
            mulQuat(mulQuat(orientationBInverse, delta), aDot);
        angularShift = (0.5 * torqueScale) * shift.yzw;
    }

    if (SceneJoint)
    {
        // Independent scene bodies may be driven kinematically without a DOF.
        // Their endpoint motion still enters the error derivative.
        linearVelocity = pointVelocity;
        angularVelocity = (0.5 * torqueScale) * mulQuat(
            mulQuat(orientationBInverse, float4(0.0, a.omega - b.omega)), a.orientation).yzw;
    }
    if (slider)
    {
        // The prismatic frame follows the owner. Its rotating axes contribute to J-dot.
        linearError = float3(slide.xy, 0.0);
        linearVelocity = slideVelocity;
        linearShift = rotateVector(-linearShift + cross(sliderAngularShift, separation) +
            cross(a.omega, 2.0 * pointVelocity - cross(a.omega, separation)), orientationAInverse);
        axialShift = linearShift.z;
    }
    else if (distance)
    {
        axialShift = dot(radial, linearShift);
        if (separationLength > 0.0)
            axialShift += (dot(pointVelocity, pointVelocity) - speed * speed) / separationLength;
    }

    const float impedancePosition = sqrt(dot(linearError, linearError) + dot(angularError, angularError));
    const bool unclamped = (config.w & (UnclampedReference | DiscreteConstraints)) != 0;
    const float3 kbi = impedanceParameters(parameters, impedancePosition, timestep, unclamped);
    const uint rowBase = id.y * limits.z + block.x;
    for (uint axis = 0; axis < linearRows; ++axis)
    {
        rows[rowBase + axis] = makeConstraintRow(uint4(block.y + axis * source.w, source.w, source.x, axis),
            EqualityRow, kbi, linearError[axis], 0.0, linearVelocity[axis], asfloat(reference.z), 0.0, linearShift[axis],
            timestep, config.w, parameters.reference.x);
    }
    for (uint axis = 0; axis < angularRows; ++axis)
    {
        const uint ordinal = axis + linearRows;
        rows[rowBase + ordinal] = makeConstraintRow(uint4(block.y + ordinal * source.w, source.w, source.x, ordinal),
            EqualityRow, kbi, angularError[axis], 0.0, angularVelocity[axis], asfloat(reference.w), 0.0,
            angularShift[axis], timestep, config.w, parameters.reference.x);
    }
    for (uint extra = 0; extra < axial.count; ++extra)
    {
        const uint ordinal = span.w + extra;
        const AxialRow row = axial.rows[extra];
        rows[rowBase + ordinal] = {uint4(block.y + ordinal * source.w, source.w, source.x, ordinal),
            float4(row.response.x, 0.0, row.direction * speed, row.response.y - row.direction * axialShift),
            row.response.z, row.response.w, row.kind};
    }
    if (swing.count != 0)
    {
        const float3 bias = rotateVector(angularShift - cross(b.omega, a.omega - b.omega), orientationBInverse);
        for (uint extra = 0; extra < swing.count; ++extra)
        {
            const uint ordinal = span.w + extra;
            const AngularRow row = swing.rows[extra];
            rows[rowBase + ordinal] = {uint4(block.y + ordinal * source.w, source.w, source.x, ordinal),
                float4(row.response.x, 0.0, dot(row.axis, relativeVelocity), row.response.y - dot(row.axis, bias)),
                row.response.z, row.response.w, row.kind};
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_spatial_equality_rows(uint3 id : SV_DispatchThreadID)
{
    emitSpatialRows<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_scene_joint_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 model = spatialModel[0];
    if (id.x >= model.x - model.z)
        return;
    emitSpatialRows<true>(id);
}

void appendSixAxisMotor(inout AngularRows result, float3 direction, float speed, float error, float targetSpeed,
                        uint mode, float4 controls, float2 spring, float weight, float timestep)
{
    const float cap = mode == 0 ? controls.z : controls.w;
    if (cap <= 0.0 || (mode == 2 && spring.x == 0.0) || (mode == 3 && all(spring == 0.0)))
        return;
    const float errorVelocity = ((mode & 1u) != 0 ? targetSpeed : 0.0) - speed;
    float diagonal = 1.0 / (weight * timestep);
    float reference = errorVelocity / timestep;
    if (mode >= 2)
    {
        diagonal = timestep * (spring.y + timestep * spring.x);
        reference = (spring.x * (error + timestep * errorVelocity) + spring.y * errorVelocity) / diagonal;
    }
    result.rows[result.count++] = {direction, float4(0.0, reference, diagonal, cap), FrictionRow};
}

AngularRows sixLinearRows(uint record, uint2 modes, float3 position, float3 velocity, float weight, float timestep)
{
    AngularRows result;
    result.count = 0;
    const float3 target = asfloat(sceneParameters[record + 13].xyz);
    const float3 targetVelocity = asfloat(sceneParameters[record + 14].xyz);
    for (uint axis = 0; axis < 3; ++axis)
    {
        float3 direction = float3(0.0);
        direction[axis] = 1.0;
        const uint motion = (modes.x >> (2 * axis)) & 3u;
        const uint motor = (modes.y >> (2 * axis)) & 3u;
        const float4 controls = asfloat(sceneParameters[record + 1 + axis]);
        const float4 springs = asfloat(sceneParameters[record + 7 + axis]);
        if (motion != 2u)
            appendAngularLimit(result, position[axis], controls.xy, direction,
                               velocity, weight, timestep, true, springs.zw);
        if (motion != 0u)
            appendSixAxisMotor(result, direction, velocity[axis], target[axis] - position[axis], targetVelocity[axis],
                              motor, controls, springs.xy, weight, timestep);
    }
    return result;
}

float swingCoordinate(float h, float part)
{
    return (h == 0.0 && part == 0.0) ? 0.0 : 2.0 * atan2(part, h);
}

float4 limitSixOrientation(float4 q, float2 twist, float2 swayY, float2 swayZ)
{
    const SwingTwistCoordinates coordinates = swingTwistCoordinates(q);
    const float2 angles = float2(swingCoordinate(coordinates.swing.x, coordinates.swing.y),
                                 swingCoordinate(coordinates.swing.x, coordinates.swing.z));
    const float2 limited = clamp(angles, float2(swayY.x, swayZ.x), float2(swayY.y, swayZ.y));
    float3 swing = coordinates.swing;
    if (any(angles != limited))
    {
        const float2 cs = cos(0.5 * limited);
        const float2 sn = sin(0.5 * limited);
        swing = normalize(float3(cs.x * cs.y, sn.x * cs.y, cs.x * sn.y));
    }
    const float angle = 0.5 * clamp(wrapTwistToRange(coordinates.twist, twist), twist.x, twist.y);
    return mulQuat(float4(swing.x, 0.0, swing.yz), float4(cos(angle), sin(angle), 0.0, 0.0));
}

float4 twistAboutAxis(float4 q, uint axis)
{
    float4 twist = float4(q.x, 0.0, 0.0, 0.0);
    twist[axis + 1] = q[axis + 1];
    return dot(twist, twist) > 0.0 ? normalize(twist) : float4(1.0, 0.0, 0.0, 0.0);
}

AngularRows sixAngularRows(uint record, uint2 modes, float4 rotation, float3 velocity, float weight, float timestep)
{
    AngularRows result;
    result.count = 0;
    const float4 controls[3] = {asfloat(sceneParameters[record + 4]), asfloat(sceneParameters[record + 5]),
                                asfloat(sceneParameters[record + 6])};
    const uint motion = modes.x >> 6u;
    const uint motors = modes.y >> 6u;
    const SwingTwistDifferential d = swingTwistDifferential(rotation, velocity, timestep);
    const SwingTwistCoordinates coordinates = swingTwistCoordinates(rotation);
    if ((motion & 3u) != 2u)
        appendAngularLimit(result, wrapTwistToRange(coordinates.twist, controls[0].xy), controls[0].xy,
                           d.twist, velocity, weight, timestep);
    const float3 derivatives[2] = {d.y, d.z};
    for (uint axis = 1; axis < 3; ++axis)
    {
        const float h = d.swing.x;
        const float part = d.swing[axis];
        const float norm = h * h + part * part;
        if (((motion >> (2 * axis)) & 3u) != 2u && norm > 1e-14)
            appendAngularLimit(result, swingCoordinate(coordinates.swing.x, coordinates.swing[axis]), controls[axis].xy,
                               (2.0 / norm) * (h * derivatives[axis - 1] - part * d.h), velocity, weight, timestep);
    }
    uint positionMask = 0u;
    for (uint axis = 0; axis < 3; ++axis)
        if (((motors >> (2 * axis)) & 3u) >= 2u && ((motion >> (2 * axis)) & 3u) != 0u)
            positionMask |= 1u << axis;
    float3 positionError = float3(0.0);
    if (positionMask != 0)
    {
        const float4 target = limitSixOrientation(asfloat(sceneParameters[record + 15]), controls[0].xy,
                                                  controls[1].xy, controls[2].xy);
        float4 delta = mulQuat(conjugateQuaternion(rotation), dot(rotation, target) < 0.0 ? -target : target);
        if (positionMask == 1u || positionMask == 2u || positionMask == 4u)
            delta = twistAboutAxis(delta, firstbitlow(positionMask));
        else if (positionMask != 7u)
            delta = mulQuat(delta, conjugateQuaternion(twistAboutAxis(delta, firstbitlow(7u ^ positionMask))));
        positionError = 2.0 * delta.yzw;
    }
    const float3 targetVelocity = asfloat(sceneParameters[record + 16].xyz);
    for (uint axis = 0; axis < 3; ++axis)
    {
        if (((motion >> (2 * axis)) & 3u) == 0u)
            continue;
        float3 basis = float3(0.0);
        basis[axis] = 1.0;
        const float3 direction = rotateVector(basis, rotation);
        appendSixAxisMotor(result, direction, dot(direction, velocity), positionError[axis], targetVelocity[axis],
                           (motors >> (2 * axis)) & 3u, controls[axis],
                           asfloat(sceneParameters[record + 10 + axis].xy), weight, timestep);
    }
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_scene_six_dof_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 model = spatialModel[0];
    if (id.x >= model.z)
        return;
    const uint record = 1 + SceneJointWords * (model.x - model.z + id.x);
    const uint4 source = spatialModel[record];
    const uint4 span = spatialModel[record + 1];
    const uint4 reference = spatialModel[record + 8];
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint bodyBase = id.y * config.x;
    const uint dofBase = id.y * config.y;
    const SpatialEndpoint a = equalityEndpoint(bodyBase, source.y, span.x, reference.x,
                                                asfloat(spatialModel[record + 4].xyz),
                                                asfloat(spatialModel[record + 6]));
    const SpatialEndpoint b = equalityEndpoint(bodyBase, source.z, span.y, reference.y,
                                                asfloat(spatialModel[record + 5].xyz),
                                                asfloat(spatialModel[record + 7]));
    const float3 delta = a.point - b.point;
    const float3 deltaVelocity = a.pointVelocity - b.pointVelocity;
    const float3 deltaOmega = a.omega - b.omega;
    const float4 inverse = conjugateQuaternion(b.orientation);
    const float3 linearVelocity = rotateVector(deltaVelocity - cross(b.omega, delta), inverse);
    const float3 angularVelocity = rotateVector(deltaOmega, inverse);
    const float timestep = asfloat(settings[2 + id.y].x);
    const uint parameters = id.y * model.y + spatialModel[record + 9].y;
    const uint2 modes = sceneParameters[parameters].xy;
    const AngularRows linear = sixLinearRows(parameters, modes, rotateVector(delta, inverse), linearVelocity,
                                            asfloat(reference.z), timestep);
    const AngularRows angular = sixAngularRows(parameters, modes, mulQuat(inverse, a.orientation), angularVelocity,
                                               asfloat(reference.w), timestep);
    if (linear.count + angular.count == 0)
        return;
    const uint2 block = reserveConstraintBlock(id.y, linear.count + angular.count, source.w);
    if (block.x == NoIndex)
        return;
    const uint columnBase = id.y * limits.w + block.y;
    float3 linearShift = float3(0.0);
    float3 angularShift = float3(0.0);
    for (uint column = 0; column < source.w; ++column)
    {
        const uint4 support = spatialModel[span.z + column];
        const bool inA = (support.y & 1u) != 0;
        const bool inB = (support.y & 2u) != 0;
        const float direction = float(inA) - float(inB);
        const uint index = dofBase + support.x;
        const SpatialMotion motion = centeredDof[index];
        const SpatialMotion derivative = support.z == NoIndex ? centeredDofDerivative[index] :
            motionCross(bodyVelocity[bodyBase + support.z], motion);
        const float3 linearColumn = rotateVector((inA ? pointJacobian(motion, a) : float3(0.0)) -
            (inB ? pointJacobian(motion, b) + cross(motion.angular.xyz, delta) : float3(0.0)), inverse);
        const float3 angularColumn = rotateVector(direction * motion.angular.xyz, inverse);
        for (uint row = 0; row < linear.count; ++row)
            columns[columnBase + row * source.w + column] = {support.x, dot(linear.rows[row].axis, linearColumn)};
        for (uint row = 0; row < angular.count; ++row)
            columns[columnBase + (linear.count + row) * source.w + column] =
                {support.x, dot(angular.rows[row].axis, angularColumn)};
        linearShift += ((inA ? pointJacobianDerivative(motion, derivative, a) : float3(0.0)) -
            (inB ? pointJacobianDerivative(motion, derivative, b) + cross(derivative.angular.xyz, delta) :
                   float3(0.0))) * qvel[index];
        angularShift += direction * derivative.angular.xyz * qvel[index];
    }
    const float3 linearBias = rotateVector(linearShift -
        cross(b.omega, 2.0 * deltaVelocity - cross(b.omega, delta)), inverse);
    const float3 angularBias = rotateVector(angularShift - cross(b.omega, deltaOmega), inverse);
    const uint rowBase = id.y * limits.z + block.x;
    for (uint ordinal = 0; ordinal < linear.count + angular.count; ++ordinal)
    {
        const bool isLinear = ordinal < linear.count;
        const AngularRow row = isLinear ? linear.rows[ordinal] : angular.rows[ordinal - linear.count];
        const float speed = dot(row.axis, isLinear ? linearVelocity : angularVelocity);
        const float bias = dot(row.axis, isLinear ? linearBias : angularBias);
        rows[rowBase + ordinal] = {uint4(block.y + ordinal * source.w, source.w, source.x, ordinal),
            float4(row.response.x, 0.0, speed, row.response.y - bias),
            row.response.z, row.response.w, row.kind};
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void write_scene_kinematic_velocities(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = sceneDimensions[0];
    if (id.x >= size.state.z)
        return;
    const BodyModel body = sceneBodies[id.x];
    const uint mocap = body.hierarchy.w;
    if (mocap == NoIndex)
        return;
    const PrescribedMotion motion = scenePrescribed[id.y * size.layout.y + mocap];
    if (motion.step != sceneFixedSteps[id.y] + 1u)
        return;
    const uint base = id.y * size.state.z;
    const float3 offset = subtreeCom[base + body.subtree.x].xyz - motion.center;
    sceneBodyVelocity[base + id.x] = spatial(motion.angular.xyz,
        motion.linear.xyz + cross(motion.angular.xyz, offset));
}
