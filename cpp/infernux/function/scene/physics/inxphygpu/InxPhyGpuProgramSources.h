// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <string_view>

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuProgramSource
{
    std::string_view module;
    bool awakeWorlds = false;
};

inline constexpr InxPhyGpuProgramSource InxPhyGpuProgramSources[] = {
    {"acceleration"},
    {"actuation"},
    {"bias"},
    {"body_controls"},
    {"broadphase"},
    {"contact_adhesion"},
    {"contact_cg"},
    {"contact_constraints"},
    {"contact_newton"},
    {"contact_reports"},
    {"convex_contacts"},
    {"discrete_constraints"},
    {"discrete_metric"},
    {"euler"},
    {"fixed_substeps"},
    {"heightfield_contacts"},
    {"implicit_free_body"},
    {"implicit"},
    {"inertia"},
    {"joint_constraints"},
    {"mass_solve"},
    {"pair_query"},
    {"particle_collision"},
    {"passive"},
    {"position"},
    {"primitive_contacts"},
    {"ray_bvh_refit"},
    {"ray_queries"},
    {"render_poses"},
    {"rk4"},
    {"scene_ccd"},
    {"shape_cast"},
    {"sleep"},
    {"spatial_constraints"},
    {"state_snapshot"},
    {"tendon_constraints"},
    {"tendon_dynamics"},
    {"tendon"},
    {"triangle_mesh_contacts"},
    {"wrench"},
    {"contact_cg", true},
    {"contact_constraints", true},
    {"contact_newton", true},
    {"mass_solve", true},
};

[[nodiscard]] ShaderTargetCode LoadInxPhyGpuProgram(std::string_view module, ShaderCodeTarget target,
                                                    bool awakeWorlds = false);

} // namespace infernux::physics::inxphygpu
