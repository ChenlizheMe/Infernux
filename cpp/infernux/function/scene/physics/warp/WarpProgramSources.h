// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <string_view>

namespace infernux::physics::warp
{

struct WarpProgramSource
{
    std::string_view module;
    bool awakeWorlds = false;
};

inline constexpr WarpProgramSource WarpProgramSources[] = {
    {"acceleration"},
    {"actuation"},
    {"bias"},
    {"broadphase"},
    {"contact_adhesion"},
    {"contact_cg"},
    {"contact_constraints"},
    {"contact_newton"},
    {"convex_contacts"},
    {"discrete_constraints"},
    {"discrete_metric"},
    {"euler"},
    {"heightfield_contacts"},
    {"implicit_free_body"},
    {"implicit"},
    {"inertia"},
    {"joint_constraints"},
    {"mass_solve"},
    {"passive"},
    {"position"},
    {"primitive_contacts"},
    {"ray_bvh_refit"},
    {"ray_queries"},
    {"rk4"},
    {"sleep"},
    {"spatial_constraints"},
    {"state_snapshot"},
    {"tendon_constraints"},
    {"tendon_dynamics"},
    {"tendon"},
    {"wrench"},
    {"contact_cg", true},
    {"contact_constraints", true},
    {"contact_newton", true},
    {"mass_solve", true},
};

[[nodiscard]] ShaderTargetCode LoadWarpProgram(std::string_view module, ShaderCodeTarget target,
                                               bool awakeWorlds = false);

} // namespace infernux::physics::warp
