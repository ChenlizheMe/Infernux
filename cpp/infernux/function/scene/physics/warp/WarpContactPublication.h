// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsContactPublication.h"
#include "WarpStep.h"

namespace infernux::physics::warp
{

/// Decode one world's accepted geometry rows against the actor and collider
/// identities captured at admission. Clones supply their own endpoint directory.
[[nodiscard]] PhysicsContactPublication::Frame
DecodeWarpContactFrame(std::vector<WarpContactReport> reports,
                       const std::vector<PhysicsContactPublication::Endpoint> &geometry,
                       const PhysicsContactPublication::Policy &policy);

} // namespace infernux::physics::warp
