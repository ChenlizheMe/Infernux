#include "PhysicsWorld.h"

#include <stdexcept>

namespace infernux
{

uint64_t PhysicsWorld::CreateRagdollAssembly(const std::vector<Rigidbody *> &, const std::vector<Joint *> &,
                                             const std::vector<int> &)
{
    throw std::logic_error("GPU CreateRagdollAssembly has not been implemented");
}

void PhysicsWorld::DestroyRagdollAssembly(uint64_t id)
{
    if (id == 0)
        return;
    throw std::logic_error("GPU DestroyRagdollAssembly has not been implemented");
}

void PhysicsWorld::DriveRagdollPose(uint64_t, const std::vector<glm::quat> &, bool)
{
    throw std::logic_error("GPU DriveRagdollPose has not been implemented");
}

} // namespace infernux
