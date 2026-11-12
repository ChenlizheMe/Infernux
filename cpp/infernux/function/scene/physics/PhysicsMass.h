#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Vec3.h>
#include <glm/vec3.hpp>

namespace JPH
{
class Shape;
}

namespace infernux
{

struct RigidbodyMassProperties;

struct PreparedBodyMass
{
    float mass;
    JPH::Vec3 inverse;
    JPH::Quat axes;
};

/// Prepare from immutable geometry or an authored tensor at constant mass.
/// Shape may be null only when authored properties supply the tensor.
PreparedBodyMass PrepareBodyMass(const JPH::Shape *shape, float mass, const RigidbodyMassProperties *properties,
                                 const glm::vec3 &scale);

} // namespace infernux
