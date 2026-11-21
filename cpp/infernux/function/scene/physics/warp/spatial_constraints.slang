// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// Slang adaptation of MuJoCo Warp _src/constraint.py connect and weld rows.

#include "model.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"
#include "prescribed_motion.slang"

static const uint SpatialEqualityWords = 9;
static const uint SceneJointWords = 11;
static const uint SceneHinge = 1;
static const uint SceneSlider = 2;
static const uint SceneDistance = 3;

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
[[vk::binding(17, 0)]] StructuredBuffer<HingeDrive> sceneDrives;

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
    const bool limited = controls.w != 0;
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

void emitSpatialRows<let SceneJoint : bool>(uint3 id)
{
    const uint4 model = spatialModel[0]; // descriptor count, equality or scene drive stride
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
    const uint4 jointType = SceneJoint ? spatialModel[record + 10] : uint4(0u);
    const bool hinge = jointType.x == SceneHinge;
    const bool slider = jointType.x == SceneSlider;
    const bool distance = jointType.x == SceneDistance;
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
    const uint4 controls = SceneJoint ? spatialModel[record + 9u] : uint4(0u);
    AxialRows axial;
    axial.count = 0;
    if (hinge)
    {
        const float theta = wrapHingeAngle(2.0 * atan2(relative.w, relative.x));
        const HingeDrive drive = sceneDrives[id.y * model.y + controls.x];
        axial = hingeRows(controls, drive, theta, speed, asfloat(reference.w), timestep, config.w, parameters);
    }
    else if (controls.w != 0)
    {
        const float2 limit = asfloat(controls.yz);
        // A range starting at zero permits its attachment points to cross.
        appendAxisLimit<true>(axial, limit, slider ? slide.z : separationLength, speed, asfloat(reference.z),
                              timestep, config.w, parameters, asfloat(jointType.yz), !distance || limit.x > 0.0);
    }
    const uint totalRows = span.w + axial.count;
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
        if (angularRows != 0u)
        {
            const float4 relativeAxis = mulQuat(mulQuat(orientationBInverse,
                float4(0.0, direction * motion.angular.xyz)), a.orientation);
            const float3 angular = (0.5 * torqueScale) * relativeAxis.yzw;
            for (uint axis = 0; axis < angularRows; ++axis)
                columns[columnBase + (axis + linearRows) * source.w + j] = {support.x, angular[axis]};
            angularVelocity += angular * speed;
            angularShift += direction * derivative.angular.xyz * speed;
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
    emitSpatialRows<true>(id);
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
