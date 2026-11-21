// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpProgramSources.h"

#include <function/renderer/shader/ShaderTargetArchive.h>

#include <iterator>
#include <stdexcept>

extern "C" const void *InfernuxWarpPhysicsArchive(unsigned program, size_t *size);

namespace infernux::physics::warp
{

ShaderTargetCode LoadWarpProgram(std::string_view module, ShaderCodeTarget target, bool awakeWorlds)
{
    for (unsigned i = 0; i < std::size(WarpProgramSources); ++i) {
        const auto &source = WarpProgramSources[i];
        if (source.module != module || source.awakeWorlds != awakeWorlds)
            continue;
        size_t size;
        const auto *bytes = static_cast<const char *>(InfernuxWarpPhysicsArchive(i, &size));
        auto code = DecodeShaderTarget({bytes, size});
        if (code.target != target)
            throw std::logic_error("Cooked Warp physics target does not match the compute device");
        return code;
    }
    throw std::invalid_argument("Unknown Warp physics program: " + std::string(module));
}

} // namespace infernux::physics::warp
