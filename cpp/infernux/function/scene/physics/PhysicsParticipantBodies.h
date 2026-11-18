#pragma once

#include "PhysicsParticipant.h"

namespace infernux
{
struct PhysicsDomain;
class JoltGpuScene;

/// Bind ordered Work ports to the actual candidate on each admitted backend.
PhysicsParticipant::RigidExchange BindPhysicsParticipantBodies(PhysicsDomain &, JoltGpuScene &, FrameWorkOwner &,
                                                               std::vector<uint32_t> bodyIds,
                                                               std::shared_ptr<std::vector<uint32_t>> gpuSlots);
} // namespace infernux
