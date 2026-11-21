// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsContactPublication.h"
#include "InxPhyGpuStep.h"

namespace infernux::physics::inxphygpu
{

/// Decode one world's accepted geometry rows against the actor and collider
/// identities captured at admission. Clones supply their own endpoint directory.
[[nodiscard]] PhysicsContactPublication::Frame
DecodeInxPhyGpuContactFrame(std::vector<InxPhyGpuContactReport> reports,
                            const std::function<PhysicsContactPublication::Endpoint(uint32_t)> &resolveGeometry,
                            const PhysicsContactPublication::Policy &policy);

} // namespace infernux::physics::inxphygpu
