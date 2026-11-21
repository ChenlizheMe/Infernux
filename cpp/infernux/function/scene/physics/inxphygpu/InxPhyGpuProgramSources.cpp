// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "InxPhyGpuProgramSources.h"

#include <function/renderer/shader/ShaderTargetArchive.h>

#include <iterator>
#include <stdexcept>

extern "C" const void *InxPhyGpuArchive(unsigned program, size_t *size);

namespace infernux::physics::inxphygpu
{

ShaderTargetCode LoadInxPhyGpuProgram(std::string_view module, ShaderCodeTarget target, bool awakeWorlds)
{
    for (unsigned i = 0; i < std::size(InxPhyGpuProgramSources); ++i) {
        const auto &source = InxPhyGpuProgramSources[i];
        if (source.module != module || source.awakeWorlds != awakeWorlds)
            continue;
        size_t size;
        const auto *bytes = static_cast<const char *>(InxPhyGpuArchive(i, &size));
        auto code = DecodeShaderTarget({bytes, size});
        if (code.target != target)
            throw std::logic_error("Cooked inxphygpu physics target does not match the compute device");
        return code;
    }
    throw std::invalid_argument("Unknown inxphygpu physics program: " + std::string(module));
}

} // namespace infernux::physics::inxphygpu
