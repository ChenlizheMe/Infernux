#pragma once

#include <function/renderer/shader/ShaderDiagnostic.h>
#include <pybind11/pybind11.h>

namespace infernux
{
pybind11::list ShaderDiagnosticsToPython(const std::vector<ShaderDiagnostic> &diagnostics);
}
