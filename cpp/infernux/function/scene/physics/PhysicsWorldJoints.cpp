#include "PhysicsWorld.h"

#include <stdexcept>

namespace infernux
{

uint64_t PhysicsWorld::CreateHingeConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                             const glm::vec3 &, const glm::vec3 &, bool, float, float, bool)
{
    throw std::logic_error("GPU CreateHingeConstraint has not been implemented");
}

uint64_t PhysicsWorld::CreateSliderConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                              const glm::vec3 &, const glm::vec3 &, bool, float, float, bool)
{
    throw std::logic_error("GPU CreateSliderConstraint has not been implemented");
}

uint64_t PhysicsWorld::CreateFixedConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                             const glm::vec3 &, bool)
{
    throw std::logic_error("GPU CreateFixedConstraint has not been implemented");
}

void PhysicsWorld::SetConstraintCollisionEnabled(uint64_t, bool)
{
    throw std::logic_error("GPU SetConstraintCollisionEnabled has not been implemented");
}

uint64_t PhysicsWorld::CreateDistanceConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                                const glm::vec3 &, bool, float, float, float, float, bool)
{
    throw std::logic_error("GPU CreateDistanceConstraint has not been implemented");
}

void PhysicsWorld::UpdateDistanceConstraintSettings(uint64_t, bool, float, float, float, float, bool)
{
    throw std::logic_error("GPU UpdateDistanceConstraintSettings has not been implemented");
}

float PhysicsWorld::GetDistanceConstraintLength(uint64_t) const
{
    throw std::logic_error("GPU GetDistanceConstraintLength has not been implemented");
}

void PhysicsWorld::UpdateHingeConstraintSettings(uint64_t, bool, float, float, bool)
{
    throw std::logic_error("GPU UpdateHingeConstraintSettings has not been implemented");
}

void PhysicsWorld::UpdateHingeConstraintMotor(uint64_t, bool, float, float, float, float, float, float, float)
{
    throw std::logic_error("GPU UpdateHingeConstraintMotor has not been implemented");
}

void PhysicsWorld::UpdateHingeConstraintTarget(uint64_t, float)
{
    throw std::logic_error("GPU UpdateHingeConstraintTarget has not been implemented");
}

void PhysicsWorld::SetHingeMotorTargets(const std::vector<HingeJoint *> &, const float *)
{
    throw std::logic_error("GPU SetHingeMotorTargets has not been implemented");
}

void PhysicsWorld::ReadHingeMotion(const std::vector<HingeJoint *> &, float *) const
{
    throw std::logic_error("GPU ReadHingeMotion has not been implemented");
}

void PhysicsWorld::UpdateSliderConstraintSettings(uint64_t, bool, float, float, bool)
{
    throw std::logic_error("GPU UpdateSliderConstraintSettings has not been implemented");
}

void PhysicsWorld::DestroyConstraintsForBody(uint32_t)
{
    // No GPU constraint can be installed until its solver is implemented.
}

void PhysicsWorld::DestroyConstraint(uint64_t constraintId)
{
    if (constraintId == 0)
        return;
    throw std::logic_error("GPU DestroyConstraint has not been implemented");
}

float PhysicsWorld::GetHingeConstraintAngle(uint64_t) const
{
    throw std::logic_error("GPU GetHingeConstraintAngle has not been implemented");
}

float PhysicsWorld::GetHingeConstraintAngularVelocity(uint64_t) const
{
    throw std::logic_error("GPU GetHingeConstraintAngularVelocity has not been implemented");
}

glm::vec2 PhysicsWorld::GetHingeConstraintMotionState(uint64_t) const
{
    throw std::logic_error("GPU GetHingeConstraintMotionState has not been implemented");
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t) const
{
    throw std::logic_error("GPU GetSliderConstraintPosition has not been implemented");
}

uint64_t PhysicsWorld::CreateSwingTwistConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                                  const SwingTwistJointSettings &, bool)
{
    throw std::logic_error("GPU CreateSwingTwistConstraint has not been implemented");
}

void PhysicsWorld::SetSwingTwistConstraintType(uint64_t, JointSwingType)
{
    throw std::logic_error("GPU SetSwingTwistConstraintType has not been implemented");
}

void PhysicsWorld::UpdateSwingTwistConstraint(uint64_t, const SwingTwistJointSettings &, bool)
{
    throw std::logic_error("GPU UpdateSwingTwistConstraint has not been implemented");
}

glm::quat PhysicsWorld::GetAngularConstraintOrientation(uint64_t) const
{
    throw std::logic_error("GPU GetAngularConstraintOrientation has not been implemented");
}

uint64_t PhysicsWorld::CreateSixDOFConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                              const SixDOFJointSettings &, bool)
{
    throw std::logic_error("GPU CreateSixDOFConstraint has not been implemented");
}

void PhysicsWorld::UpdateSixDOFConstraint(uint64_t, const SixDOFJointSettings &, bool)
{
    throw std::logic_error("GPU UpdateSixDOFConstraint has not been implemented");
}

glm::vec3 PhysicsWorld::GetSixDOFConstraintPosition(uint64_t) const
{
    throw std::logic_error("GPU GetSixDOFConstraintPosition has not been implemented");
}

} // namespace infernux
