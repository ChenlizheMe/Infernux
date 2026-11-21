#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "JoltGpuShape.h"
#include "PhysicsShapeAuthoring.h"
#include "PhysicsTargetReference.h"
#include <function/scene/physics/gpu/SleepTypes.h>

namespace infernux
{

/// Owner-thread creation data. Only initialState is an initial condition;
/// subsequent authored model changes must retain the resident GPU state.
struct PreparedGpuActor
{
    JPH::GPU::BodyModel body;
    JPH::GPU::BodySleepAuthor sleep;
    PreparedGpuShape geometry;
    JPH::GPU::BodyState initialState;
    JPH::GPU::Float4 rootScale{1, 1, 1, 0};
    PhysicsActorPublication publication;
    JPH::RefConst<JPH::Shape> authoredShape;
    struct Member
    {
        PhysicsTargetReference target;
        uint64_t colliderId, generation, gameObjectId;
    };
    // Ordinals come from the authored outer compound.
    std::vector<Member> members;

    void SetOriginPose(const glm::vec3 &position, const glm::quat &rotation);
};

[[nodiscard]] PreparedGpuActor PrepareJoltGpuActor(PhysicsActorAuthoring actor, uint32_t bodySlot, uint32_t worldIndex);

} // namespace infernux
