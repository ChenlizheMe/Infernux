#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/MassProperties.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "../RigidbodyMassProperties.h"
#include "PhysicsMass.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infernux
{

namespace
{

static JPH::Mat44 ScaledAuthoredInertia(const RigidbodyMassProperties &properties, const glm::vec3 &scale)
{
    const glm::dmat3 rotation = glm::mat3_cast(glm::dquat(properties.rotation));
    glm::dmat3 diagonal(0.0);
    for (int i = 0; i < 3; ++i)
        diagonal[i][i] = properties.moments[i];
    const glm::dmat3 inertia = rotation * diagonal * glm::transpose(rotation);
    const double halfTrace = .5 * (inertia[0][0] + inertia[1][1] + inertia[2][2]);
    glm::dmat3 secondMoment(0.0);
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            secondMoment[column][row] =
                ((column == row ? halfTrace : 0.0) - inertia[column][row]) * double(scale[column]) * double(scale[row]);
    const double trace = secondMoment[0][0] + secondMoment[1][1] + secondMoment[2][2];
    JPH::Mat44 result = JPH::Mat44::sIdentity();
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            result(row, column) = float((column == row ? trace : 0.0) - secondMoment[column][row]);
    // Constant authored mass: do not multiply by volume and divide it out.
    // This also retains valid planar inertia at a single zero scale axis.
    return result;
}

static PreparedBodyMass PrepareMassConfiguration(const JPH::MassProperties &properties)
{
    const auto diagonal = properties.mInertia.GetDiagonal3();
    const float magnitude = std::max({diagonal.GetX(), diagonal.GetY(), diagonal.GetZ()});
    if (!std::isfinite(magnitude) || magnitude <= 0.0f)
        throw std::invalid_argument("rigidbody inertia is not finite positive at the current scale");
    auto normalized = properties;
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            normalized.mInertia(row, column) = float(double(properties.mInertia(row, column)) / magnitude);
    JPH::Mat44 axes;
    JPH::Vec3 moments;
    if (!normalized.DecomposePrincipalMomentsOfInertia(axes, moments))
        throw std::invalid_argument("rigidbody inertia cannot be decomposed at the current scale");
    JPH::Vec3 inverse = JPH::Vec3::sZero();
    for (int i = 0; i < 3; ++i) {
        const double moment = double(moments[i]) * magnitude;
        const float value = float(1.0 / moment);
        if (moment <= 0.0 || !std::isfinite(value) || value <= 0.0f)
            throw std::invalid_argument("rigidbody inverse inertia is not representable at the current scale");
        inverse.SetComponent(i, value);
    }
    return {properties.mMass, inverse, axes.GetQuaternion()};
}

} // namespace

PreparedBodyMass PrepareBodyMass(const JPH::Shape *shape, float mass, const RigidbodyMassProperties *properties,
                                 const glm::vec3 &scale)
{
    JPH::MassProperties value;
    if (properties) {
        value.mMass = mass;
        value.mInertia = ScaledAuthoredInertia(*properties, scale);
    } else {
        value = shape->GetMassProperties();
        value.ScaleToMass(mass);
    }
    return PrepareMassConfiguration(value);
}

} // namespace infernux
