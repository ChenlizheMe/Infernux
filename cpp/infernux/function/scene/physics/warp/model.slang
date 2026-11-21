// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/types.py and _src/math.py.

static const uint JointFree = 0;
static const uint JointBall = 1;
static const uint JointSlide = 2;
static const uint JointHinge = 3;
static const uint NoIndex = 0xffffffffu;
static const float PhysicsPi = 3.141592653589793;

struct ModelDimensions
{
    uint4 state; // nq, nv, nbody, njnt
    uint4 layout; // ngeom, nmocap, nbranch, nM
};

struct BodyModel
{
    uint4 hierarchy; // parent, first joint, joint count, mocap row
    uint4 subtree; // root, subtree body count, output branch, unused
    float4 position;
    float4 rotation;
    float4 inertialPosition;
    float4 inertialRotation;
    float4 inertiaMass; // principal inertia xyz, mass
    float4 subtreeMass; // subtree mass, gravity compensation scale, unused
    uint4 dofs; // first dof, dof count, unused
};

struct JointModel
{
    uint4 indices; // type, qpos address, dof address, body
    float4 anchor;
    float4 axis; // local axis xyz, reference position for scalar joints
};

struct GeomModel
{
    uint4 indices; // body, geom type, collision type, collision affinity
    float4 position;
    float4 rotation;
};

struct DofModel
{
    uint4 indices; // body, parent dof, sparse row address, nonzero count
    float4 armature;
};

struct Pose
{
    float4 position;
    float4 rotation; // MuJoCo quaternion order wxyz
};

struct JointPose
{
    float4 anchor;
    float4 axis;
};

struct CartesianFrame
{
    float4 position;
    float4 row0;
    float4 row1;
    float4 row2;
};

struct SpatialMotion
{
    float4 angular;
    float4 linear;
};

struct SpatialInertia
{
    float4 rotation; // Ixx, Iyy, Izz, Ixy
    float4 products; // Ixz, Iyz, unused
    float4 firstMomentMass;
};

float4 mulQuat(float4 left, float4 right)
{
    return float4(left.x * right.x - dot(left.yzw, right.yzw),
                  left.x * right.yzw + right.x * left.yzw + cross(left.yzw, right.yzw));
}

float3 rotateVector(float3 vectorValue, float4 rotation)
{
    const float s = rotation.x;
    const float3 u = rotation.yzw;
    float3 rotated = 2.0 * (dot(u, vectorValue) * u) + (s * s - dot(u, u)) * vectorValue;
    return rotated + 2.0 * s * cross(u, vectorValue);
}

float4 axisAngleQuat(float3 axis, float angle)
{
    float sine, cosine;
    sincos(angle * 0.5, sine, cosine);
    return float4(cosine, axis * sine);
}

float3 rotationVector(float4 rotation)
{
    const float sine = length(rotation.yzw);
    if (sine == 0.0)
        return float3(0.0);
    float angle = 2.0 * atan2(sine, rotation.x);
    if (angle > PhysicsPi)
        angle -= 2.0 * PhysicsPi;
    return rotation.yzw * (angle / sine);
}

float4 integrateQuat(float4 rotation, float3 velocity, float dt)
{
    const float speed = length(velocity);
    const float3 axis = speed > 0.0 ? velocity / speed : float3(0.0);
    return normalize(mulQuat(normalize(rotation), axisAngleQuat(axis, dt * speed)));
}

CartesianFrame cartesianFrame(float3 position, float4 rotation)
{
    const float w = rotation.x, x = rotation.y, y = rotation.z, z = rotation.w;
    CartesianFrame frame;
    frame.position = float4(position, 0.0);
    frame.row0 = float4(w*w + x*x - y*y - z*z, 2.0*(x*y - w*z), 2.0*(x*z + w*y), 0.0);
    frame.row1 = float4(2.0*(x*y + w*z), w*w - x*x + y*y - z*z, 2.0*(y*z - w*x), 0.0);
    frame.row2 = float4(2.0*(x*z - w*y), 2.0*(y*z + w*x), w*w - x*x - y*y + z*z, 0.0);
    return frame;
}

SpatialMotion spatial(float3 angular, float3 linear)
{
    SpatialMotion result;
    result.angular = float4(angular, 0.0);
    result.linear = float4(linear, 0.0);
    return result;
}

SpatialMotion inertiaTimesMotion(SpatialInertia inertia, SpatialMotion motion)
{
    const float3 angular = motion.angular.xyz;
    const float3 moment = inertia.firstMomentMass.xyz;
    float3 rotation = float3(
        dot(float3(inertia.rotation.x, inertia.rotation.w, inertia.products.x), angular),
        dot(float3(inertia.rotation.w, inertia.rotation.y, inertia.products.y), angular),
        dot(float3(inertia.products.x, inertia.products.y, inertia.rotation.z), angular));
    return spatial(rotation + cross(moment, motion.linear.xyz),
                   cross(angular, moment) + inertia.firstMomentMass.w * motion.linear.xyz);
}

float spatialDot(SpatialMotion left, SpatialMotion right)
{
    return dot(left.angular.xyz, right.angular.xyz) + dot(left.linear.xyz, right.linear.xyz);
}

void addMotion(inout SpatialMotion sum, SpatialMotion term, float scale)
{
    sum.angular += scale * term.angular;
    sum.linear += scale * term.linear;
}

SpatialMotion motionCross(SpatialMotion left, SpatialMotion right)
{
    return spatial(cross(left.angular.xyz, right.angular.xyz),
                   cross(left.linear.xyz, right.angular.xyz) + cross(left.angular.xyz, right.linear.xyz));
}

SpatialMotion motionCrossForce(SpatialMotion motion, SpatialMotion force)
{
    return spatial(cross(motion.angular.xyz, force.angular.xyz) + cross(motion.linear.xyz, force.linear.xyz),
                   cross(motion.angular.xyz, force.linear.xyz));
}
